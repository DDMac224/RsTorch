use num_traits::NumOps;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use std::{fmt::Debug, io, sync::Arc};

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
        }
    }
}
