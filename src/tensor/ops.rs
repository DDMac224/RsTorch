use crate::tensor::{Device, Element, FloatElement, Tensor};

mod cpu_ops;
mod cuda_ops;

macro_rules! elemwise_bin_op_impl {
    ($method:ident; $cpu_fn:ident, $cuda_fn:ident, $scalar_op:ident) => {
        impl<T> Tensor<T>
        where
            T: Element,
        {
            pub(crate) fn $method(&self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.device(),
                    rhs.device(),
                    "Tensors must be on the same device"
                );

                match self.device() {
                    Device::CPU => self.$cpu_fn(&rhs, T::$scalar_op),
                    Device::Cuda => self.$cuda_fn(&rhs, T::$scalar_op),
                }
            }
        }
    };
}

elemwise_bin_op_impl!(elemwise_add;  cpu_elemwise_bin, cuda_elemwise_bin, add);
elemwise_bin_op_impl!(elemwise_sub; cpu_elemwise_bin, cuda_elemwise_bin, sub);
elemwise_bin_op_impl!(elemwise_mul; cpu_elemwise_bin, cuda_elemwise_bin, mul);
elemwise_bin_op_impl!(elemwise_div; cpu_elemwise_bin, cuda_elemwise_bin, div);

macro_rules! elemwise_uni_op_impl {
    ($method:ident, $generic_trait:ident; $cpu_fn:ident, $cuda_fn:ident, $scalar_op:ident) => {
        impl<T> Tensor<T>
        where
            T: $generic_trait,
        {
            pub(crate) fn $method(&self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.device(),
                    rhs.device(),
                    "Tensors must be on the same device"
                );

                match self.device() {
                    Device::CPU => self.$cpu_fn(T::$scalar_op),
                    Device::Cuda => self.$cuda_fn(T::$scalar_op),
                }
            }
        }
    };
}

elemwise_uni_op_impl!(elemwise_sin, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, sin);
elemwise_uni_op_impl!(elemwise_cos, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, cos);
elemwise_uni_op_impl!(elemwise_tan, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, tan);
elemwise_uni_op_impl!(elemwise_asin, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, asin);
elemwise_uni_op_impl!(elemwise_acos, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, acos);
elemwise_uni_op_impl!(elemwise_atan, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, atan);
elemwise_uni_op_impl!(elemwise_sinh, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, sinh);
elemwise_uni_op_impl!(elemwise_cosh, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, cosh);
elemwise_uni_op_impl!(elemwise_tanh, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, tanh);
elemwise_uni_op_impl!(elemwise_asinh, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, asinh);
elemwise_uni_op_impl!(elemwise_acosh, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, acosh);
elemwise_uni_op_impl!(elemwise_atanh, FloatElement; cpu_elemwise_uni, cuda_elemwise_uni, atanh);

macro_rules! tensor_op_impl {
    ($method:ident; $cpu_fn:ident, $cuda_fn:ident) => {
        impl<T> Tensor<T>
        where
            T: Element,
        {
            pub(crate) fn $method(&self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.device(),
                    rhs.device(),
                    "Tensors must be on the same device"
                );

                match self.device() {
                    Device::CPU => self.$cpu_fn(&rhs),
                    Device::Cuda => self.$cuda_fn(&rhs),
                }
            }
        }
    };
}

tensor_op_impl!(forward_matmul; cpu_matmul, cuda_matmul);

#[cfg(test)]
mod tests {}
