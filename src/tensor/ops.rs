use std::{
    ops::{Add, AddAssign, Sub, SubAssign},
    sync::Arc,
};

use crate::tensor::{Device, Element, Tensor};

mod cpu_ops;
mod cuda_ops;

impl<T> Tensor<T>
where
    T: Element,
{
    fn index<I: AsRef<[usize]>>(&self, idx: I) -> Self {
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

macro_rules! elementwise_op_impl {
    ($Trait:ident, $method:ident; $TraitAssign:ident,$method_assign:ident; $cpu_fn:ident, $cuda_fn:ident) => {
        impl<T> $Trait<&Tensor<T>> for &Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape(), "Dimension mismatch");
                assert_eq!(
                    self.device(),
                    rhs.device(),
                    "Tensors must be on the same device"
                );

                match self.device() {
                    Device::CPU => self.$cpu_fn(&rhs, T::$method),
                    Device::Cuda => self.$cuda_fn(&rhs, T::$method),
                }
            }
        }

        impl<T> $Trait<Tensor<T>> for Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        impl<T> $Trait<&Tensor<T>> for Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl<T> $Trait<Tensor<T>> for &Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl<T> $TraitAssign<&Tensor<T>> for Tensor<T>
        where
            T: Element,
        {
            fn $method_assign(&mut self, rhs: &Tensor<T>) {
                *self = (&*self).$method(rhs);
            }
        }
        impl<T> $TraitAssign<Tensor<T>> for Tensor<T>
        where
            T: Element,
        {
            fn $method_assign(&mut self, rhs: Tensor<T>) {
                self.$method_assign(&rhs)
            }
        }
    };
}

elementwise_op_impl!(Add, add; AddAssign, add_assign; cpu_elemwise, cpu_elemwise);
elementwise_op_impl!(Sub, sub; SubAssign, sub_assign; cpu_elemwise, cpu_elemwise);

#[cfg(test)]
mod tests {}
