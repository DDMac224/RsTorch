use crate::tensor::{Device, Element, Tensor};

mod cpu_ops;
mod cuda_ops;

macro_rules! elemwise_op_impl {
    ($method:ident; $cpu_fn:ident, $cuda_fn:ident, $scalar_op:ident) => {
        impl<T> Tensor<T>
        where
            T: Element,
        {
            pub fn $method(&self, rhs: &Tensor<T>) -> Tensor<T> {
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

elemwise_op_impl!(elemwise_add;  cpu_elemwise, cuda_elemwise, add);
elemwise_op_impl!(elemwise_sub; cpu_elemwise, cuda_elemwise, add);
elemwise_op_impl!(elemwise_mul; cpu_elemwise, cuda_elemwise, mul);
elemwise_op_impl!(elemwise_div; cpu_elemwise, cuda_elemwise, div);

macro_rules! tensor_op_impl {
    ($method:ident; $cpu_fn:ident, $cuda_fn:ident) => {
        impl<T> Tensor<T>
        where
            T: Element,
        {
            pub fn $method(&self, rhs: &Tensor<T>) -> Tensor<T> {
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

tensor_op_impl!(mul; cpu_matmul, cuda_matmul);

#[cfg(test)]
mod tests {}
