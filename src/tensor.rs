use num_traits::{NumOps, ToPrimitive};
use rand::{
    Fill, Rng,
    distr::{Distribution, StandardUniform},
};
use std::{fmt::Debug, io, ops::Index};

pub mod ops;

pub trait Element: NumOps + Copy + PartialEq + Debug {}
impl<T: NumOps + Copy + PartialEq + Debug> Element for T {}

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Element> {
    data: Vec<T>,
    strides: Vec<usize>,
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

        let mut strides: Vec<usize> = Vec::new();
        let mut stride = 1;
        for dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();

        Ok(Self {
            data: data,
            strides: strides,
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
}
