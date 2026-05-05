mod common;

use common::to_vec;
use rstorch::tensor::{Device, Tensor};

// =========================================================================
// Tensor::new
// =========================================================================

#[test]
fn test_new_1d() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], None, None);
    assert_eq!(t.shape(), vec![4]);
    assert_eq!(to_vec(&t), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_new_2d() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None);
    assert_eq!(t.shape(), vec![2, 2]);
    assert_eq!(to_vec(&t), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_new_3d() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = Tensor::new(data, vec![2, 3, 4], None, None);
    assert_eq!(t.shape(), vec![2, 3, 4]);
    assert_eq!(t.size(), 24);
}

// =========================================================================
// Tensor::zeros
// =========================================================================

#[test]
fn test_zeros() {
    let t = Tensor::<f32>::zeros(vec![3, 4], None, None);
    assert_eq!(t.shape(), vec![3, 4]);
    assert_eq!(to_vec(&t), vec![0.0f32; 12]);
}

// =========================================================================
// Tensor::ones
// =========================================================================

#[test]
fn test_ones() {
    let t = Tensor::<f32>::ones(vec![2, 3], None, None);
    assert_eq!(t.shape(), vec![2, 3]);
    assert_eq!(to_vec(&t), vec![1.0f32; 6]);
}

// =========================================================================
// Tensor::zeros_like
// =========================================================================

#[test]
fn test_zeros_like() {
    let t = Tensor::new(vec![5.0f32, 6.0, 7.0], vec![3], None, None);
    let z = Tensor::zeros_like(&t, None);
    assert_eq!(z.shape(), t.shape());
    assert_eq!(to_vec(&z), vec![0.0f32; 3]);
}

// =========================================================================
// Tensor::ones_like
// =========================================================================

#[test]
fn test_ones_like() {
    let t = Tensor::new(vec![5.0f32, 6.0, 7.0], vec![3], None, None);
    let o = Tensor::ones_like(&t, None);
    assert_eq!(o.shape(), vec![3]);
    assert_eq!(to_vec(&o), vec![1.0f32; 3]);
}

// =========================================================================
// Tensor::new_rand
// =========================================================================

#[test]
fn test_new_rand_shape() {
    let t = Tensor::<f32>::new_rand(vec![4, 5], None, None);
    assert_eq!(t.shape(), vec![4, 5]);
    assert_eq!(t.size(), 20);
}

// =========================================================================
// Device
// =========================================================================

#[test]
fn test_device_default_cpu() {
    let t = Tensor::new(vec![1.0f32], vec![1], None, None);
    assert_eq!(t.device(), Device::CPU);
}

#[test]
#[should_panic]
fn test_device_explicit_cuda_panics() {
    let t = Tensor::new(vec![1.0f32], vec![1], Some(Device::Cuda), None);
    assert_eq!(t.device(), Device::Cuda);
}

// =========================================================================
// Tensor::item
// =========================================================================

#[test]
fn test_item_scalar() {
    let t = Tensor::new(vec![42.0f32], vec![1], None, None);
    assert_eq!(t.item().unwrap(), 42.0);
}

#[test]
fn test_item_multi_element_errors() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    assert!(t.item().is_err());
}

// =========================================================================
// Tensor::index
// =========================================================================

#[test]
fn test_index_1d() {
    let t = Tensor::new(vec![10.0f32, 20.0, 30.0], vec![3], None, None);
    assert_eq!(t.index([0]).item().unwrap(), 10.0);
    assert_eq!(t.index([1]).item().unwrap(), 20.0);
    assert_eq!(t.index([2]).item().unwrap(), 30.0);
}

#[test]
fn test_index_2d() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None, None);
    assert_eq!(t.index([0, 0]).item().unwrap(), 1.0);
    assert_eq!(t.index([0, 2]).item().unwrap(), 3.0);
    assert_eq!(t.index([1, 0]).item().unwrap(), 4.0);
    assert_eq!(t.index([1, 2]).item().unwrap(), 6.0);
}

// =========================================================================
// Tensor::reshape
// =========================================================================

#[test]
fn test_reshape_2d_to_1d() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None, None);
    let r = t.reshape(vec![6]);
    assert_eq!(r.shape(), vec![6]);
    assert_eq!(to_vec(&r), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_reshape_1d_to_2d() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], None, None);
    let r = t.reshape(vec![2, 3]);
    assert_eq!(r.shape(), vec![2, 3]);
    assert_eq!(to_vec(&r), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_reshape_2d_to_2d() {
    let t = Tensor::new(vec![1.0f32; 12], vec![3, 4], None, None);
    let r = t.reshape(vec![2, 6]);
    assert_eq!(r.shape(), vec![2, 6]);
    assert_eq!(r.size(), 12);
}

#[test]
#[should_panic]
fn test_reshape_incompatible_panics() {
    let t = Tensor::new(vec![0.0f32; 6], vec![2, 3], None, None);
    t.reshape(vec![4, 2]);
}

// =========================================================================
// Tensor::broadcast_to
// =========================================================================

#[test]
fn test_broadcast_to_scalar_to_vector() {
    let t = Tensor::new(vec![7.0f32], vec![1], None, None);
    let b = t.broadcast_to(&vec![4]);
    assert_eq!(b.shape(), vec![4]);
    assert_eq!(to_vec(&b), vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn test_broadcast_to_row_to_matrix() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![1, 3], None, None);
    let b = t.broadcast_to(&vec![2, 3]);
    assert_eq!(b.shape(), vec![2, 3]);
    assert_eq!(to_vec(&b), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_broadcast_to_same_shape() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    let b = t.broadcast_to(&vec![3]);
    assert_eq!(b.shape(), vec![3]);
    assert_eq!(to_vec(&b), vec![1.0, 2.0, 3.0]);
}

#[test]
#[should_panic]
fn test_broadcast_to_incompatible_panics() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    t.broadcast_to(&vec![2, 2]);
}

// =========================================================================
// Tensor::broadcast_tensors
// =========================================================================

#[test]
fn test_broadcast_tensors_same_shape() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    let b = Tensor::new(vec![4.0f32, 5.0, 6.0], vec![3], None, None);
    let (ba, bb) = a.broadcast_tensors(&b);
    assert_eq!(ba.shape(), vec![3]);
    assert_eq!(bb.shape(), vec![3]);
}

#[test]
fn test_broadcast_tensors_scalar_to_vector() {
    let scalar = Tensor::new(vec![5.0f32], vec![1], None, None);
    let vec3 = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
    let (bs, bv) = scalar.broadcast_tensors(&vec3);
    assert_eq!(bs.shape(), vec![3]);
    assert_eq!(bv.shape(), vec![3]);
}
