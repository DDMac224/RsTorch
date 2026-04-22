use itertools::{self, Itertools};
use num_traits::NumOps;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use std::{cmp, fmt::Debug, io, sync::Arc};

pub mod ops;

pub trait Element: NumOps + Copy + PartialEq + Debug {}
impl<T: NumOps + Copy + PartialEq + Debug> Element for T {}

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Element> {
    data: Arc<Vec<T>>,
    stride: Vec<usize>,
    shape: Vec<usize>,
    device: Device,
    offset: usize,
    broadcasted_shape: Option<Vec<usize>>,
    broadcasted_stride: Option<Vec<usize>>,
    broadcasted_data_cycles: Option<usize>,
}

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>, device: Option<Device>) -> Tensor<T> {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Length of data does not match shape"
        );

        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        Self {
            data: Arc::new(data),
            stride,
            shape: shape,
            device: device.unwrap_or(Device::CPU),
            offset: 0,
            broadcasted_shape: None,
            broadcasted_data_cycles: None,
            broadcasted_stride: None,
        }
    }

    pub fn new_rand(shape: Vec<usize>, device: Option<Device>) -> Tensor<T>
    where
        StandardUniform: Distribution<T>,
    {
        let rng = rand::rng();
        let num_elems = shape.iter().product();
        let rand_data: Vec<T> = rng.random_iter().take(num_elems).collect();

        Self::new(rand_data, shape, device)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn is_contiguous(&self) -> bool {
        // stride will never be empty
        self.stride[self.stride.len()] == 1 as usize
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> Result<(), io::Error> {
        if shape.iter().product::<usize>() != self.shape().iter().product() {
            return Err(io::Error::new(io::ErrorKind::Other, ""));
        }

        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        self.shape = shape;
        self.stride = stride;

        Ok(())
    }

    fn is_broadcastable(&self, other: &Self) -> bool {
        self.shape()
            .iter()
            .zip(other.shape().iter())
            .rev()
            .all(|(&s1, &s2)| s1 == s2 || s1 == 1 || s2 == 1)
    }

    pub fn broadcast(&mut self, other: &mut Self) {
        assert!(self.is_broadcastable(other), "not broadcastable");

        let mut new_shape: Vec<usize> = Vec::new();

        let mut self_strd: Vec<usize> = Vec::new();
        let mut other_strd: Vec<usize> = Vec::new();

        for dim in self.shape.iter().zip_longest(other.shape().iter()).rev() {
            match dim {
                itertools::EitherOrBoth::Both(self_dim, other_dim) => {
                    new_shape.push(cmp::max(*self_dim, *other_dim));
                    match (self_dim, other_dim) {
                        (1, 1) => {
                            self_strd.push(usize::MAX);
                            other_strd.push(usize::MAX);
                        }
                        (1, _) => {
                            self_strd.push(0);
                            other_strd.push(usize::MAX);
                        }
                        (_, 1) => {
                            self_strd.push(usize::MAX);
                            other_strd.push(0);
                        }
                        _ => {
                            self_strd.push(usize::MAX);
                            other_strd.push(usize::MAX);
                        }
                    }
                }
                itertools::EitherOrBoth::Left(self_dim) => {
                    new_shape.push(*self_dim);
                    self_strd.push(usize::MAX);
                    other_strd.push(0);
                }
                itertools::EitherOrBoth::Right(other_dim) => {
                    new_shape.push(*other_dim);
                    self_strd.push(0);
                    other_strd.push(usize::MAX);
                }
            }
        }

        new_shape.reverse();
        let mut self_broadcasted_stride = self.stride();
        let mut other_broadcasted_stride = other.stride();

        if self.stride.len() < self_strd.len() {
            let mut pad = vec![0; self_strd.len() - self.stride.len()];
            pad.extend(self.stride());
            self_broadcasted_stride = pad;
        }
        if other.stride.len() < other_strd.len() {
            let mut pad = vec![0; other_strd.len() - other.stride().len()];
            pad.extend(other.stride());
            other_broadcasted_stride = pad;
        }

        self_broadcasted_stride = self_broadcasted_stride
            .iter()
            .zip(self_strd)
            .map(|(strd, mask)| strd & mask)
            .collect();
        other_broadcasted_stride = other_broadcasted_stride
            .iter()
            .zip(other_strd)
            .map(|(strd, mask)| strd & mask)
            .collect();

        self.broadcasted_stride = Some(self_broadcasted_stride);
        self.broadcasted_shape = Some(new_shape.clone());

        other.broadcasted_stride = Some(other_broadcasted_stride);
        other.broadcasted_shape = Some(new_shape);
    }

    pub fn item(&self) -> Result<T, io::Error> {
        if self.shape.len() != 1 || self.shape[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "A Tensor with multiple elements cannot return a scalar.",
            ));
        }

        Ok(self.data[self.offset])
    }

    pub fn index<I: AsRef<[usize]>>(&self, idx: I) -> Self {
        let index = idx.as_ref();

        assert!(
            index.len() <= self.shape().len(),
            "Indexing too many dimensions."
        );
        assert!(
            !self.shape.iter().zip(index.iter()).any(|(dim, i)| i >= dim),
            "Index out of bounds."
        );

        let new_offset: usize = self
            .stride
            .iter()
            .zip(index.iter())
            .map(|(strd, i)| strd * i)
            .sum::<usize>()
            + self.offset;
        let new_shape = &self.shape()[self.shape.len() - index.len()..];
        let new_stride = &self.stride()[self.stride.len() - index.len()..];

        Self {
            data: Arc::clone(&self.data),
            stride: Vec::from(new_stride),
            shape: Vec::from(new_shape),
            device: self.device(),
            offset: new_offset,
            broadcasted_shape: None,
            broadcasted_data_cycles: None,
            broadcasted_stride: None,
        }
    }
}
