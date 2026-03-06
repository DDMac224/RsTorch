use crate::tensor::{Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn cpu_elemwise(&self, rhs: &Self, op: fn(T, T) -> T) -> Self {
        todo!()
    }
}
