use crate::tensor::{Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn cuda_elemwise_bin(&self, rhs: &Self, op: fn(T, T) -> T) -> Self {
        todo!()
    }

    pub fn cuda_elemwise_uni(&self, op: fn(T) -> T) -> Self {
        todo!()
    }

    pub fn cuda_matmul(&self, rhs: &Self) -> Self {
        todo!()
    }
}
