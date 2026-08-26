use num_traits::{Float, NumOps, One, Zero};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use std::{
    fmt::Debug,
    io,
    iter::repeat,
    ops::{Deref, DerefMut},
    sync::{Arc, OnceLock, RwLock},
};

use crate::{
    autograd::node::GradNode,
    tensor::{data::TensorData, metadata::TensorMetadata},
};

pub mod broadcast;
pub mod cpu;
pub mod cuda;
pub mod data;
pub mod grad;
pub mod metadata;
pub mod ops;
pub mod transform;

pub trait Element: NumOps + Zero + One + Copy + PartialEq + Debug + 'static {}
impl<T: NumOps + Zero + One + Copy + PartialEq + Debug + 'static> Element for T {}

pub trait FloatElement: Element + Float {}

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Element>(Arc<TensorInner<T>>);

#[derive(Debug, Clone)]
pub struct TensorInner<T: Element> {
    data: Arc<RwLock<TensorData<T>>>,
    metadata: TensorMetadata,
    pub(crate) grad_node: OnceLock<GradNode<T>>,
    requires_grad: bool,
}

impl<T: Element> Deref for Tensor<T> {
    type Target = TensorInner<T>;

    fn deref(&self) -> &TensorInner<T> {
        &self.0
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn new(
        data: Vec<T>,
        shape: Vec<usize>,
        device: Option<Device>,
        requires_grad: Option<bool>,
    ) -> Tensor<T> {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Length of data does not match shape"
        );

        let grad_node = OnceLock::new();
        let _ = grad_node.set(GradNode::leaf());

        Self(Arc::new(TensorInner {
            data: Arc::new(RwLock::new(TensorData::new(
                data,
                device.unwrap_or(Device::CPU),
            ))),
            grad_node,
            requires_grad: requires_grad.unwrap_or(false),
            metadata: TensorMetadata::new(shape),
        }))
    }

    pub(crate) fn from_parts(
        data: TensorData<T>,
        metadata: TensorMetadata,
        grad_node: Option<GradNode<T>>,
        requires_grad: bool,
    ) -> Self {
        assert_eq!(
            data.len(),
            metadata.size(),
            "Length of data does not match shape"
        );

        let set_node = OnceLock::new();
        if let Some(node) = grad_node {
            let _ = set_node.set(node);
        }

        Self(Arc::new(TensorInner {
            data: Arc::new(RwLock::new(data)),
            metadata,
            grad_node: set_node,
            requires_grad,
        }))
    }

    pub fn new_rand(
        shape: Vec<usize>,
        device: Option<Device>,
        requires_grad: Option<bool>,
    ) -> Tensor<T>
    where
        StandardUniform: Distribution<T>,
    {
        let rng = rand::rng();
        let num_elems = shape.iter().product();
        let rand_data: Vec<T> = rng.random_iter().take(num_elems).collect();

        Self::new(rand_data, shape, device, requires_grad)
    }

    pub fn zeros(
        shape: Vec<usize>,
        device: Option<Device>,
        requires_grad: Option<bool>,
    ) -> Tensor<T> {
        let num_elems = shape.iter().product();
        let zeros: Vec<T> = repeat(T::zero()).take(num_elems).collect();

        Self::new(zeros, shape, device, requires_grad)
    }

    pub fn zeros_like(t: &Tensor<T>, requires_grad: Option<bool>) -> Tensor<T> {
        Self::zeros(t.shape(), Some(t.device()), requires_grad)
    }

    pub fn ones(
        shape: Vec<usize>,
        device: Option<Device>,
        requires_grad: Option<bool>,
    ) -> Tensor<T> {
        let num_elems = shape.iter().product();
        let ones: Vec<T> = repeat(T::one()).take(num_elems).collect();

        Self::new(ones, shape, device, requires_grad)
    }

    pub fn ones_like(t: &Tensor<T>, requires_grad: Option<bool>) -> Tensor<T> {
        Self::ones(t.shape(), Some(t.device()), requires_grad)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.metadata.shape.clone()
    }

    pub fn stride(&self) -> Vec<usize> {
        self.metadata.stride.clone()
    }

    pub fn size(&self) -> usize {
        self.metadata.size()
    }

    pub fn rank(&self) -> usize {
        self.metadata.rank()
    }

    pub fn device(&self) -> Device {
        match *self.data.read().unwrap() {
            TensorData::CpuData(_) => Device::CPU,
            TensorData::CudaData => Device::Cuda,
        }
    }

    pub fn item(&self) -> Result<T, io::Error> {
        if !self.metadata.is_scalar() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "A Tensor with multiple elements cannot return a scalar.",
            ));
        }

        Ok(self.data.read().unwrap().item(&self.metadata))
    }

    pub fn index<I: AsRef<[usize]>>(&self, idx: I) -> Self {
        let index = idx.as_ref();

        Self(Arc::new(TensorInner {
            data: Arc::clone(&self.data),
            metadata: self.metadata.index(index),
            grad_node: OnceLock::new(),
            requires_grad: self.requires_grad,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Tensor::new — stride internals
    // =========================================================================

    #[test]
    fn test_new_1d_stride() {
        let t = Tensor::new(vec![10i32, 20, 30], vec![3], None, None);
        assert_eq!(t.shape(), vec![3]);
        assert_eq!(t.stride(), vec![1]);
    }

    #[test]
    fn test_new_3d_stride() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::new(data, vec![2, 3, 4], None, None);
        assert_eq!(t.stride(), vec![12, 4, 1]);
    }

    #[test]
    #[should_panic(expected = "Length of data does not match shape")]
    fn test_new_shape_mismatch_panics() {
        Tensor::new(vec![1.0f32, 2.0], vec![3], None, None);
    }

    #[test]
    fn test_new_device_default_is_cpu() {
        let t = Tensor::new(vec![1u8], vec![1], None, None);
        assert_eq!(t.device(), Device::CPU);
    }

    #[test]
    #[ignore]
    fn test_new_device_explicit_cuda() {
        let t = Tensor::new(vec![1u8], vec![1], Some(Device::Cuda), None);
        assert_eq!(t.device(), Device::Cuda);
    }

    #[test]
    fn test_new_rand_total_elements() {
        let t = Tensor::<f64>::new_rand(vec![3, 3, 3], None, None);
        assert_eq!(t.stride()[0], 9);
    }

    // =========================================================================
    // contiguous
    // =========================================================================

    #[test]
    fn test_contiguous_already_contiguous_unchanged() {
        let data: Vec<i32> = (0..6).collect();
        let t = Tensor::new(data.clone(), vec![2, 3], None, None);
        assert_eq!(t.shape(), vec![2, 3]);
        for i in 0..6 {
            assert_eq!(t.index([i / 3, i % 3]).item().unwrap(), data[i]);
        }
    }

    #[test]
    fn test_contiguous_resets_offset() {
        let data: Vec<i32> = (0..6).collect();
        let t = Tensor::new(data, vec![2, 3], None, None);
        let row = t.index([1]);
        let row_cont = row.contiguous();
        assert_eq!(row_cont.metadata.offset, 0);
    }

    #[test]
    fn test_contiguous_broadcast_stride_materializes_correctly() {
        let t = Tensor::new(vec![1i32, 2, 3], vec![1, 3], None, None);
        let b = t.broadcast_to(&vec![2, 3]).contiguous();
        assert_eq!(b.index([0, 0]).item().unwrap(), 1);
        assert_eq!(b.index([0, 1]).item().unwrap(), 2);
        assert_eq!(b.index([0, 2]).item().unwrap(), 3);
        assert_eq!(b.index([1, 0]).item().unwrap(), 1);
        assert_eq!(b.index([1, 1]).item().unwrap(), 2);
        assert_eq!(b.index([1, 2]).item().unwrap(), 3);
    }

    #[test]
    fn test_contiguous_is_contiguous_after_broadcast() {
        let t = Tensor::new(vec![1i32, 2, 3], vec![1, 3], None, None);
        let b = t.broadcast_to(&vec![2, 3]);
        assert!(!b.is_contiguous());
        let b_cont = b.contiguous();
        assert!(b_cont.is_contiguous());
    }

    #[test]
    fn test_contiguous_index_math_correctness() {
        let data: Vec<i32> = (0..24).collect();
        let t = Tensor::new(data, vec![2, 3, 4], None, None);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(
                        t.index([i, j, k]).item().unwrap(),
                        (i * 12 + j * 4 + k) as i32,
                        "Mismatch at [{i},{j},{k}]"
                    );
                }
            }
        }
    }

    // =========================================================================
    // is_contiguous
    // =========================================================================

    #[test]
    fn test_is_contiguous_fresh_tensor() {
        let t = Tensor::new(vec![0f32; 6], vec![2, 3], None, None);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_1d() {
        let t = Tensor::new(vec![1, 2, 3, 4], vec![4], None, None);
        assert!(t.is_contiguous());
    }

    // =========================================================================
    // index — panic cases (internal bound checks)
    // =========================================================================

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds_panics() {
        let t = Tensor::new(vec![1, 2, 3], vec![3], None, None);
        t.index([5]);
    }

    #[test]
    #[should_panic(expected = "Indexing too many dimensions")]
    fn test_index_too_many_dims_panics() {
        let t = Tensor::new(vec![1, 2, 3], vec![3], None, None);
        t.index([0, 1]);
    }

    // =========================================================================
    // broadcast_tensors — different ranks (internal logic)
    // =========================================================================

    #[test]
    fn test_broadcast_tensors_different_ranks() {
        let a = Tensor::new(vec![0f32; 3], vec![1, 3], None, None);
        let b = Tensor::new(vec![0f32; 6], vec![2, 1, 3], None, None);
        let (ba, bb) = a.broadcast_tensors(&b);
        assert_eq!(ba.shape(), vec![2, 1, 3]);
        assert_eq!(bb.shape(), vec![2, 1, 3]);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_tensors_incompatible_panics() {
        let a = Tensor::new(vec![0f32; 2], vec![2], None, None);
        let b = Tensor::new(vec![0f32; 3], vec![3], None, None);
        a.broadcast_tensors(&b);
    }

    // =========================================================================
    // Device equality
    // =========================================================================

    #[test]
    fn test_device_equality() {
        assert_eq!(Device::CPU, Device::CPU);
        assert_eq!(Device::Cuda, Device::Cuda);
        assert_ne!(Device::CPU, Device::Cuda);
    }

    // =========================================================================
    // Arc shared data (clone does not copy underlying data)
    // =========================================================================

    #[test]
    fn test_clone_shares_data() {
        let t = Tensor::new(vec![1, 2, 3, 4], vec![2, 2], None, None);
        let t2 = t.clone();
        assert_eq!(t.shape(), t2.shape());
        assert_eq!(t.stride(), t2.stride());
    }
}
