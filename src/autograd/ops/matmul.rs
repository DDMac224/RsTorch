use std::sync::Arc;

use crate::{
    autograd::{BackwardFn, ops::op_impl},
    tensor::{Element, Tensor},
};

#[derive(Debug)]
pub struct MatMulBackward<T: Element> {
    lhs: Arc<Tensor<T>>,
    rhs: Arc<Tensor<T>>,
}

impl<T> BackwardFn<T> for MatMulBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>) {
        self.lhs
            .backward(Arc::new(grad_output.forward_matmul(&self.rhs.transpose())));
        self.rhs
            .backward(Arc::new(self.lhs.transpose().forward_matmul(&grad_output)));
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
    fn matmul_grad(&self, rhs: &Self) -> Self {
        let ret = self.forward_matmul(rhs);

        let backwrd_fn: Arc<dyn BackwardFn<T>> = Arc::new(MatMulBackward {
            lhs: Arc::new(self.clone()),
            rhs: Arc::new(rhs.clone()),
        });

        if self.requires_grad() || rhs.requires_grad() {
            ret.set_grad_fn(Arc::clone(&backwrd_fn));
        }

        ret
    }
}

op_impl!(matmul; matmul_grad, MatMulBackward);
