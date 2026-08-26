use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::tensor::{Element, Tensor};

pub mod node;
pub mod ops;

pub trait BackwardFn<T>: Debug + DynClone
where
    T: Element,
{
    fn backward(&self, grad_output: Tensor<T>);
}

dyn_clone::clone_trait_object!(<T> BackwardFn<T> where T: Element);
