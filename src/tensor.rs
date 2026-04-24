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

    pub fn contiguous(&self) {
        let size: usize = self.shape.iter().product();
        let mut data: Vec<T> = Vec::new();

        for i in 0..size {
            data.push(
                self.data[self.offset
                    + self
                        .shape
                        .iter()
                        .rev()
                        .scan(i, |acc, e| {
                            let temp = *acc;
                            *acc /= *e;
                            Some(e % temp)
                        })
                        .zip(self.stride.iter())
                        .fold(0, |acc, (idx, strd)| acc + (idx * strd))],
            );
        }
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expt_strd: usize = 1;

        for (&strd, &shp) in self.stride.iter().zip(self.shape.iter().rev()) {
            if strd != expt_strd {
                return false;
            }
            expt_strd *= shp;
        }
        return true;
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

    fn is_broadcastable(&self, target: &Vec<usize>) -> bool {
        self.shape()
            .iter()
            .zip(target.iter())
            .rev()
            .all(|(&ss, &ts)| ss == ts || ss == 1)
            && self.shape.len() <= target.len()
    }

    pub fn broadcast_to(&self, target: &Vec<usize>) -> Self {
        assert!(
            self.is_broadcastable(target),
            "Tensor of shape: {:?} cannot be broadcasted to shape: {:?}",
            self.shape,
            target
        );
        let mut strd_mask: Vec<usize> = Vec::new();

        for dim in self.shape.iter().zip_longest(target.iter()).rev() {
            match dim {
                itertools::EitherOrBoth::Both(self_dim, target_dim) => {
                    match (self_dim, target_dim) {
                        (1, 1) => {
                            strd_mask.push(usize::MAX);
                        }
                        (1, _) => {
                            strd_mask.push(0);
                        }
                        _ => {
                            strd_mask.push(usize::MAX);
                        }
                    }
                }
                itertools::EitherOrBoth::Left(_) => {
                    panic!("Tensor is shorter than target");
                }
                itertools::EitherOrBoth::Right(_) => {
                    strd_mask.push(0);
                }
            }
        }
        let mut self_broadcasted_stride = self.stride();

        if self.stride.len() < strd_mask.len() {
            let mut pad = vec![0; strd_mask.len() - self.stride.len()];
            pad.extend(self.stride());
            self_broadcasted_stride = pad;
        }

        self_broadcasted_stride = self_broadcasted_stride
            .iter()
            .zip(strd_mask)
            .map(|(strd, mask)| strd & mask)
            .collect();

        Self {
            data: Arc::clone(&self.data),
            stride: self_broadcasted_stride,
            shape: target.clone(),
            device: self.device(),
            offset: self.offset.clone(),
        }
    }

    pub fn broadcast_tensors(&self, other: &Self) -> (Self, Self) {
        let mut new_shape: Vec<usize> = Vec::new();

        for dim in self.shape.iter().zip_longest(other.shape().iter()).rev() {
            match dim {
                itertools::EitherOrBoth::Both(self_dim, other_dim) => {
                    new_shape.push(cmp::max(*self_dim, *other_dim));
                }
                itertools::EitherOrBoth::Left(self_dim) => {
                    new_shape.push(*self_dim);
                }
                itertools::EitherOrBoth::Right(other_dim) => {
                    new_shape.push(*other_dim);
                }
            }
        }

        new_shape.reverse();

        assert!(
            self.is_broadcastable(&new_shape) && other.is_broadcastable(&new_shape),
            "Tensor of shape: {:?} and Tensor of shape: {:?} are not broadcastable.",
            self.shape,
            other.shape()
        );

        (
            self.broadcast_to(&new_shape),
            other.broadcast_to(&new_shape),
        )
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
            .fold(0, |acc, (strd, i)| acc + strd * i)
            + self.offset;
        let new_shape = &self.shape()[self.shape.len() - index.len()..];
        let new_stride = &self.stride()[self.stride.len() - index.len()..];

        Self {
            data: Arc::clone(&self.data),
            stride: Vec::from(new_stride),
            shape: Vec::from(new_shape),
            device: self.device(),
            offset: new_offset,
        }
    }
}
