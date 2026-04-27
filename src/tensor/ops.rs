use std::{
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
    sync::Arc,
};

use crate::tensor::{Device, Element, Tensor};

mod cpu_ops;
mod cuda_ops;

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

impl<T> Mul<&Tensor<T>> for &Tensor<T>
where
    T: Element,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(
            self.device(),
            rhs.device(),
            "Tensors must be on the same device"
        );

        match self.device() {
            Device::CPU => self.cpu_matmul(rhs),
            Device::Cuda => panic!("unimplemented"),
        }
    }
}

#[cfg(test)]
mod tests {}
