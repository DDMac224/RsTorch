use crate::{
    autograd::{BackwardFn, ops::op_impl},
    tensor::{Element, Tensor},
};

use std::{ops::Add, sync::Arc};

#[derive(Debug)]
pub struct AddBackward<T: Element> {
    lhs: Arc<Tensor<T>>,
    rhs: Arc<Tensor<T>>,
}

impl<T> BackwardFn<T> for AddBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>) {
        self.lhs.backward(Arc::clone(&self.lhs));
        self.rhs.backward(Arc::clone(&self.rhs));
    }

    fn zero_grad(&self) {
        self.lhs.zero_grad();
        self.rhs.zero_grad();
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    fn add_grad(&self, rhs: &Self) -> Self {
        let ret = self.elemwise_add(rhs);

        let backwrd_fn: Arc<dyn BackwardFn<T>> = Arc::new(AddBackward {
            lhs: Arc::new(self.clone()),
            rhs: Arc::new(rhs.clone()),
        });

        if self.requires_grad() || rhs.requires_grad() {
            ret.set_grad_fn(Arc::clone(&backwrd_fn));
        }

        ret
    }
}

op_impl!(Add, add; add_grad, AddBackward);
