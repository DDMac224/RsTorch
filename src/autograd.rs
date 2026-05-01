use std::{fmt::Debug, sync::Arc};

use crate::tensor::{Element, Tensor};

pub mod ops;
pub trait BackwardFn<T>: Debug
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>);
    fn zero_grad(&self);
}
