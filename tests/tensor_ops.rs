mod common;

use common::to_vec;
use rstorch::tensor::Tensor;

// =========================================================================
// Element-wise Add
// =========================================================================

#[test]
fn test_add_1d() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    let b = Tensor::new(vec![4.0f32, 5.0, 6.0], vec![3], None, None);
    let c = &a + &b;
    assert_eq!(to_vec(&c), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_add_2d() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None);
    let b = Tensor::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2], None, None);
    let c = &a + &b;
    assert_eq!(to_vec(&c), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_add_broadcast_scalar() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    let b = Tensor::new(vec![10.0f32], vec![1], None, None);
    let c = &a + &b;
    assert_eq!(to_vec(&c), vec![11.0, 12.0, 13.0]);
}

#[test]
fn test_add_owned() {
    let a = Tensor::new(vec![1.0f32, 2.0], vec![2], None, None);
    let b = Tensor::new(vec![3.0f32, 4.0], vec![2], None, None);
    let c = a + b;
    assert_eq!(to_vec(&c), vec![4.0, 6.0]);
}

// =========================================================================
// Element-wise Sub
// =========================================================================

#[test]
fn test_sub_1d() {
    let a = Tensor::new(vec![10.0f32, 20.0, 30.0], vec![3], None, None);
    let b = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    let c = &a - &b;
    assert_eq!(to_vec(&c), vec![9.0, 18.0, 27.0]);
}

#[test]
fn test_sub_2d() {
    let a = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], None, None);
    let b = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None);
    let c = &a - &b;
    assert_eq!(to_vec(&c), vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_sub_broadcast_scalar() {
    let a = Tensor::new(vec![10.0f32, 20.0, 30.0], vec![3], None, None);
    let b = Tensor::new(vec![5.0f32], vec![1], None, None);
    let c = &a - &b;
    assert_eq!(to_vec(&c), vec![5.0, 15.0, 25.0]);
}

#[test]
fn test_sub_owned() {
    let a = Tensor::new(vec![5.0f32, 10.0], vec![2], None, None);
    let b = Tensor::new(vec![2.0f32, 3.0], vec![2], None, None);
    let c = a - b;
    assert_eq!(to_vec(&c), vec![3.0, 7.0]);
}

// =========================================================================
// Matrix Multiplication
// =========================================================================

#[test]
fn test_matmul_1d_dot_product() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![1, 3], None, None);
    let b = Tensor::new(vec![4.0f32, 5.0, 6.0], vec![3, 1], None, None);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), vec![1, 1]);
    assert!((c.item().unwrap() - 32.0).abs() < 1e-5);
}

#[test]
fn test_matmul_2x2() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None);
    let b = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], None, None);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(to_vec(&c), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_2x3_by_3x2() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None, None);
    let b = Tensor::new(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], None, None);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(to_vec(&c), vec![58.0, 64.0, 139.0, 154.0]);
}
