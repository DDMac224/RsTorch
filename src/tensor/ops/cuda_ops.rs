use crate::tensor::{Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    fn cuda_elemwise(&self, rhs: &Self, op: fn(T, T) -> T) -> Self {
        todo!()
    }
}
