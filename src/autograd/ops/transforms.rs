use std::sync::Arc;

use crate::{
    autograd::BackwardFn,
    tensor::{Element, Tensor},
};

#[derive(Debug, Clone)]
pub(crate) struct IndexFn<T>
where
    T: Element,
{
    parent: Arc<Tensor<T>>,
}

impl<T> BackwardFn<T> for IndexFn<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Tensor<T>) {
        todo!()
    }
}
