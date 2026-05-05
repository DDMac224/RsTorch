mod common;

use common::to_vec;
use rstorch::tensor::Tensor;

// =========================================================================
// Autograd: Forward Pass
// =========================================================================

#[test]
fn test_add_requires_grad_propagates() {
    let a = Tensor::new(vec![1.0f32, 2.0], vec![2], None, Some(true));
    let b = Tensor::new(vec![3.0f32, 4.0], vec![2], None, Some(false));
    let c = &a + &b;
    assert!(c.requires_grad());
}

#[test]
fn test_sub_requires_grad_propagates() {
    let a = Tensor::new(vec![5.0f32, 6.0], vec![2], None, Some(false));
    let b = Tensor::new(vec![1.0f32, 2.0], vec![2], None, Some(true));
    let c = &a - &b;
    assert!(c.requires_grad());
}

#[test]
fn test_matmul_requires_grad_propagates() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, Some(true));
    let b = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], None, Some(true));
    let c = a.matmul(&b);
    assert!(c.requires_grad());
}

#[test]
fn test_no_grad_when_neither_requires_grad() {
    let a = Tensor::new(vec![1.0f32, 2.0], vec![2], None, Some(false));
    let b = Tensor::new(vec![3.0f32, 4.0], vec![2], None, Some(false));
    let c = &a + &b;
    assert!(c.requires_grad());
}

// =========================================================================
// Autograd: Backward Pass (Gradient Computation)
// =========================================================================

#[test]
fn test_backward_add_gradients() {
    let a = Tensor::new(vec![1.0f32, 2.0], vec![2], None, Some(true));
    let b = Tensor::new(vec![3.0f32, 4.0], vec![2], None, Some(true));
    let c = &a + &b;
    c.backward();

    let a_grad = a.grad().expect("a should have gradient");
    let b_grad = b.grad().expect("b should have gradient");

    assert_eq!(to_vec(&a_grad), vec![1.0, 1.0]);
    assert_eq!(to_vec(&b_grad), vec![1.0, 1.0]);
}

#[test]
fn test_backward_sub_gradients() {
    let a = Tensor::new(vec![5.0f32, 10.0], vec![2], None, Some(true));
    let b = Tensor::new(vec![2.0f32, 3.0], vec![2], None, Some(true));
    let c = &a - &b;
    c.backward();

    let a_grad = a.grad().expect("a should have gradient");
    let b_grad = b.grad().expect("b should have gradient");

    assert_eq!(to_vec(&a_grad), vec![1.0, 1.0]);
    assert_eq!(to_vec(&b_grad), vec![-1.0, -1.0]);
}

#[test]
fn test_backward_matmul_gradients() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, Some(true));
    let b = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], None, Some(true));
    let c = a.matmul(&b);
    c.backward();

    let a_grad = a.grad().expect("a should have gradient");
    let b_grad = b.grad().expect("b should have gradient");

    assert_eq!(to_vec(&a_grad), vec![11.0, 15.0, 11.0, 15.0]);
    assert_eq!(to_vec(&b_grad), vec![4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_backward_chain() {
    let a = Tensor::new(vec![1.0f32, 2.0], vec![2], None, Some(true));
    let b = Tensor::new(vec![3.0f32, 4.0], vec![2], None, Some(true));
    let c = &a + &b;
    let d = &c - &a;
    d.backward();

    let a_grad = a.grad().expect("a should have gradient");
    let b_grad = b.grad().expect("b should have gradient");

    assert_eq!(to_vec(&a_grad), vec![0.0, 0.0]);
    assert_eq!(to_vec(&b_grad), vec![1.0, 1.0]);
}

#[test]
fn test_backward_scalar_result() {
    let a = Tensor::new(vec![2.0f32], vec![1], None, Some(true));
    let b = Tensor::new(vec![3.0f32], vec![1], None, Some(true));
    let c = &a + &b;
    c.backward();

    let a_grad = a.grad().expect("a should have gradient");
    let b_grad = b.grad().expect("b should have gradient");

    assert_eq!(to_vec(&a_grad), vec![1.0]);
    assert_eq!(to_vec(&b_grad), vec![1.0]);
}
