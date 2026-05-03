use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use dyn_clone::DynClone;

use crate::tensor::{Element, Tensor};

pub mod node;
pub mod ops;

pub trait BackwardFn<T>: Debug + DynClone
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>);
    fn zero_grad(&self);
}

dyn_clone::clone_trait_object!(<T> BackwardFn<T> where T:Element);
