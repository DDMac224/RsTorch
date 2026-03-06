use num_traits::NumOps;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use std::{fmt::Debug, io};

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
    data: Vec<T>,
    stride: Vec<usize>,
    shape: Vec<usize>,
    device: Device,
}

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn new(
        data: Vec<T>,
        shape: Vec<usize>,
        device: Option<Device>,
    ) -> Result<Tensor<T>, io::Error> {
        if data.len() != shape.iter().product() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Length of data does not match shape",
            ));
        }

        let mut stride: Vec<usize> = Vec::new();
        let mut s = 1;
        for dim in shape.iter().rev() {
            stride.push(s);
            s *= dim;
        }
        stride.reverse();

        Ok(Self {
            data: data,
            stride,
            shape: shape,
            device: device.unwrap_or(Device::CPU),
        })
    }

    pub fn new_rand(shape: Vec<usize>, device: Option<Device>) -> Result<Tensor<T>, io::Error>
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
        todo!()
    }
}
