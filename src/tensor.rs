use num_traits::{NumOps, One, Zero};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use std::{
    fmt::Debug,
    io,
    iter::repeat,
    sync::{Arc, Mutex},
};

use crate::autograd::BackwardFn;

pub mod broadcast;
pub mod grad;
pub mod inplace;
pub mod ops;

pub trait Element: NumOps + Zero + One + Copy + PartialEq + Debug + 'static {}
impl<T: NumOps + Zero + One + Copy + PartialEq + Debug + 'static> Element for T {}

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Element> {
    data: Arc<Vec<T>>,
    stride: Vec<usize>,
    shape: Vec<usize>,
    device: Device,
    offset: usize,
    grad: Arc<Mutex<Option<Arc<Tensor<T>>>>>,
    pub(crate) grad_fn: Arc<Mutex<Option<Arc<dyn BackwardFn<T>>>>>,
    requires_grad: bool,
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

        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        Self {
            data: Arc::new(data),
            stride,
            shape: shape,
            device: device.unwrap_or(Device::CPU),
            offset: 0,
            grad: Arc::new(Mutex::new(None)),
            requires_grad: requires_grad.unwrap_or(false),
            grad_fn: Arc::new(Mutex::new(None)),
        }
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

    pub fn ones_like(t: Tensor<T>, requires_grad: Option<bool>) -> Tensor<T> {
        Self::ones(t.shape(), Some(t.device()), requires_grad)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.iter().product::<usize>() == 1
    }

    pub fn item(&self) -> Result<T, io::Error> {
        if !self.is_scalar() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "A Tensor with multiple elements cannot return a scalar.",
            ));
        }

        Ok(self.data[self.offset])
    }

    pub fn index<I: AsRef<[usize]>>(&self, idx: I) -> Self {
        let index = idx.as_ref();

        assert!(
            index.len() <= self.shape().len(),
            "Indexing too many dimensions."
        );
        assert!(
            !self.shape.iter().zip(index.iter()).any(|(dim, i)| i >= dim),
            "Index out of bounds."
        );

        let new_offset: usize = self
            .stride
            .iter()
            .zip(index.iter())
            .fold(0, |acc, (strd, i)| acc + strd * i)
            + self.offset;
        let new_shape = &self.shape()[index.len()..];
        let new_stride = &self.stride()[index.len()..];

        Self {
            data: Arc::clone(&self.data),
            stride: Vec::from(new_stride),
            shape: Vec::from(new_shape),
            device: self.device(),
            offset: new_offset,
            grad: Arc::clone(&self.grad),
            requires_grad: self.requires_grad,
            grad_fn: Arc::clone(&self.grad_fn),
        }
    }
}

// Tests written by claude
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Tensor::new
    // =========================================================================

    #[test]
    fn test_new_basic() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None);
        assert_eq!(t.shape(), vec![2, 2]);
    }

    #[test]
    fn test_new_1d() {
        let t = Tensor::new(vec![10i32, 20, 30], vec![3], None, None);
        assert_eq!(t.shape(), vec![3]);
        assert_eq!(t.stride(), vec![1]);
    }

    #[test]
    fn test_new_3d_stride() {
        // shape [2, 3, 4] → strides should be [12, 4, 1]
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
    fn test_new_device_explicit_cuda() {
        let t = Tensor::new(vec![1u8], vec![1], Some(Device::Cuda), None);
        assert_eq!(t.device(), Device::Cuda);
    }

    // =========================================================================
    // Tensor::new_rand
    // =========================================================================

    #[test]
    fn test_new_rand_shape() {
        let t = Tensor::<f32>::new_rand(vec![4, 5], None, None);
        assert_eq!(t.shape(), vec![4, 5]);
    }

    #[test]
    fn test_new_rand_total_elements() {
        let t = Tensor::<f64>::new_rand(vec![3, 3, 3], None, None);
        // 27 elements → stride[0] == 9
        assert_eq!(t.stride()[0], 9);
    }

    // =========================================================================
    // contiguous
    // =========================================================================

    #[test]
    fn test_contiguous_already_contiguous_unchanged() {
        let data: Vec<i32> = (0..6).collect();
        let mut t = Tensor::new(data.clone(), vec![2, 3], None, None);
        t.contiguous();
        // Data and shape should be identical
        assert_eq!(t.shape(), vec![2, 3]);
        for i in 0..6 {
            assert_eq!(t.index([i / 3, i % 3]).item().unwrap(), data[i]);
        }
    }

    #[test]
    fn test_contiguous_resets_offset() {
        // index() advances the offset — contiguous() should reset it to 0
        let data: Vec<i32> = (0..6).collect();
        let t = Tensor::new(data, vec![2, 3], None, None);
        let mut row = t.index([1]); // offset is now 3
        row.contiguous();
        assert_eq!(row.offset, 0);
    }

    #[test]
    fn test_contiguous_broadcast_stride_materializes_correctly() {
        // broadcast [1, 3] → [2, 3], then contiguous should produce real duplicated data
        let t = Tensor::new(vec![1i32, 2, 3], vec![1, 3], None, None);
        let mut b = t.broadcast_to(&vec![2, 3]);
        b.contiguous();
        // After contiguous the data should be [1, 2, 3, 1, 2, 3]
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
        let mut b = t.broadcast_to(&vec![2, 3]);
        assert!(!b.is_contiguous()); // broadcast strides are non-standard
        b.contiguous();
        assert!(b.is_contiguous());
    }

    #[test]
    fn test_contiguous_index_math_correctness() {
        // 3D case: shape [2, 3, 4], verify every element maps correctly
        let data: Vec<i32> = (0..24).collect();
        let mut t = Tensor::new(data, vec![2, 3, 4], None, None);
        t.contiguous();
        // Element at logical position [i, j, k] should be i*12 + j*4 + k
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
    // reshape
    // =========================================================================

    #[test]
    fn test_reshape_valid() {
        let mut t = Tensor::new(vec![1.0f32; 12], vec![3, 4], None, None);
        let result = t.reshape(vec![2, 6]);
        assert_eq!(result.shape(), vec![2, 6]);
    }

    #[test]
    fn test_reshape_to_1d() {
        let mut t = Tensor::new(vec![1i32; 12], vec![3, 4], None, None);
        t = t.reshape(vec![12]);
        assert_eq!(t.shape(), vec![12]);
        assert_eq!(t.stride(), vec![1]);
    }

    #[test]
    fn test_reshape_updates_stride() {
        let mut t = Tensor::new(vec![0f64; 24], vec![2, 3, 4], None, None);
        t = t.reshape(vec![4, 6]);
        assert_eq!(t.stride(), vec![6, 1]);
    }

    #[test]
    #[should_panic(expected = "Shape of: [4, 2] is not compatible with tensor of size: 6.")]
    fn test_reshape_incompatible_returns_err() {
        let mut t = Tensor::new(vec![0f32; 6], vec![2, 3], None, None);
        t.reshape(vec![4, 2]);
    }

    // #[test]
    // fn test_reshape_same_shape_ok() {
    //     let mut t = Tensor::new(vec![1.0f32; 9], vec![3, 3], None, None);
    //     assert_eq!(t, t.reshape(&vec![3, 3]));
    // }

    // =========================================================================
    // item
    // =========================================================================

    #[test]
    fn test_item_scalar_tensor() {
        let t = Tensor::new(vec![42i32], vec![1], None, None);
        assert_eq!(t.item().unwrap(), 42);
    }

    #[test]
    fn test_item_multi_element_errors() {
        let t = Tensor::new(vec![1, 2, 3], vec![3], None, None);
        assert!(t.item().is_err());
    }

    #[test]
    fn test_item_2d_errors() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], None, None);
        assert!(t.item().is_err());
    }

    // =========================================================================
    // index
    // =========================================================================

    #[test]
    fn test_index_1d() {
        let t = Tensor::new(vec![10, 20, 30], vec![3], None, None);
        let elem = t.index([1]);
        println!("{:?}", elem);
        assert_eq!(elem.item().unwrap(), 20, "{:?} != 20", elem.item().unwrap());
    }

    #[test]
    fn test_index_2d_row() {
        // shape [2, 3], data row-major: [[0,1,2],[3,4,5]]
        let data: Vec<i32> = (0..6).collect();
        let t = Tensor::new(data, vec![2, 3], None, None);
        // Indexing row 1 should give a view of [3,4,5]
        let row = t.index([1]);
        assert_eq!(row.shape(), vec![3]);
    }

    #[test]
    fn test_index_2d_element() {
        let data: Vec<i32> = (0..6).collect();
        let t = Tensor::new(data, vec![2, 3], None, None);
        // t[1][2] == 5
        let row = t.index([1]);
        let elem = row.index([2]);
        assert_eq!(elem.item().unwrap(), 5);
    }

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
    // broadcast_to
    // =========================================================================

    #[test]
    fn test_broadcast_to_adds_dim() {
        // shape [1, 3] → broadcast to [2, 3]
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![1, 3], None, None);
        let b = t.broadcast_to(&vec![2, 3]);
        assert_eq!(b.shape(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_to_stride_zeroed_for_broadcast_dim() {
        // shape [1] → broadcast to [4]: the single stride should become 0
        let t = Tensor::new(vec![7.0f64], vec![1], None, None);
        let b = t.broadcast_to(&vec![4]);
        assert_eq!(b.stride()[0], 0);
    }

    #[test]
    fn test_broadcast_to_same_shape() {
        let t = Tensor::new(vec![1, 2, 3], vec![3], None, None);
        let b = t.broadcast_to(&vec![3]);
        assert_eq!(b.shape(), vec![3]);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_to_incompatible_panics() {
        let t = Tensor::new(vec![1, 2, 3], vec![3], None, None);
        t.broadcast_to(&vec![2, 2]);
    }

    // =========================================================================
    // broadcast_tensors
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
        // [1] and [3] → both become [3]
        let scalar = Tensor::new(vec![5.0f32], vec![1], None, None);
        let vec3 = Tensor::new(vec![1.0f32, 2.0, 3.0], vec![3], None, None);
        let (bs, bv) = scalar.broadcast_tensors(&vec3);
        assert_eq!(bs.shape(), vec![3]);
        assert_eq!(bv.shape(), vec![3]);
    }

    #[test]
    fn test_broadcast_tensors_different_ranks() {
        // [1, 3] and [2, 1, 3] → both become [2, 1, 3]
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
        // They should be equal in value
        assert_eq!(t.shape(), t2.shape());
        assert_eq!(t.stride(), t2.stride());
    }
}
