use std::sync::{Arc, Mutex};

use crate::{
    autograd::BackwardFn,
    tensor::{Element, Tensor},
};

#[derive(Debug)]
pub(crate) struct GradNode<T: Element> {
    pub(crate) grad: Mutex<Option<Arc<Tensor<T>>>>,
    pub(crate) grad_fn: Option<Box<dyn BackwardFn<T>>>,
}

impl<T: Element> Clone for GradNode<T> {
    fn clone(&self) -> Self {
        GradNode {
            grad: Mutex::new(None),
            grad_fn: self.grad_fn.clone(),
        }
    }
}

impl<T: Element> GradNode<T> {
    pub fn leaf() -> Self {
        Self {
            grad: Mutex::new(None),
            grad_fn: None,
        }
    }

    pub fn new(grad_fn: Box<dyn BackwardFn<T>>) -> Self {
        Self {
            grad: Mutex::new(None),
            grad_fn: Some(grad_fn),
        }
    }

    pub fn set_grad(&self, grad: Arc<Tensor<T>>) {
        *self.grad.lock().unwrap() = Some(grad.clone());
    }

    pub fn update_grad(&self, grad: Arc<Tensor<T>>) {
        let mut grad_mut = self.grad.lock().unwrap();

        if let Some(ref mut mut_t) = *grad_mut {
            *mut_t = Arc::new(mut_t.elemwise_add(&grad));
        } else {
            *grad_mut = Some(grad.clone());
        }
    }

    pub fn zero_grad(&self) {
        todo!()
    }

    pub fn backward(&self) {
        todo!()
    }

    pub fn call_backwardfn(&self, grad_output: Arc<Tensor<T>>) {}
}
