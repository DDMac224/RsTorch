use std::ops::Index;

use crate::tensor::{Element, Tensor};

impl<T, I> Index<I> for Tensor<T>
where
    T: Element,
    I: AsRef<[usize]>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        let index = index.as_ref();

        if index.len() != self.shape.len() {
            panic!("Shape of index does not match tensor shape.");
        }

        if self.shape.iter().zip(index.iter()).any(|(dim, i)| i >= dim) {
            panic!("Index out of bounds.")
        }

        println!(
            "{:?}",
            self.strides
                .iter()
                .zip(index.iter())
                .collect::<Vec<(&usize, &usize)>>()
        );
        &self.data[self
            .strides
            .iter()
            .zip(index.iter())
            .map(|(stride, index)| stride * index)
            .sum::<usize>()]
    }
}

#[cfg(test)]
mod tests {

    use crate::tensor::Tensor;

    use super::*;

    // Tests written by claude
    #[test]
    fn index() {
        let t1 = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], None).unwrap();

        // Test all elements with different index types
        // Row 0: [1, 2, 3]
        assert_eq!(t1[vec![0, 0]], 1);
        assert_eq!(t1[&[0, 1]], 2);
        assert_eq!(t1[[0, 2]], 3);

        // Row 1: [4, 5, 6]
        assert_eq!(t1[vec![1, 0]], 4);
        assert_eq!(t1[&[1, 1]], 5);
        assert_eq!(t1[[1, 2]], 6);

        // Test 3D tensor
        let t2 = Tensor::new(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            vec![2, 2, 3],
            None,
        )
        .unwrap();

        // Shape [2, 2, 3] means:
        // [[[1, 2, 3],
        //   [4, 5, 6]],
        //  [[7, 8, 9],
        //   [10, 11, 12]]]

        assert_eq!(t2[[0, 0, 0]], 1);
        assert_eq!(t2[vec![0, 0, 2]], 3);
        assert_eq!(t2[&[0, 1, 1]], 5);
        assert_eq!(t2[[1, 0, 0]], 7);
        assert_eq!(t2[vec![1, 1, 2]], 12);

        // Test 1D tensor
        let t3 = Tensor::new(vec![10, 20, 30], vec![3], None).unwrap();
        assert_eq!(t3[&[0]], 10);
        assert_eq!(t3[[1]], 20);
        assert_eq!(t3[vec![2]], 30);
    }

    #[test]
    #[should_panic(expected = "Shape of index does not match tensor shape")]
    fn index_wrong_dimensions() {
        let t1 = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], None).unwrap();
        let _ = t1[&[0]]; // Should panic - need 2 indices, not 1
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn index_out_of_bounds() {
        let t1 = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], None).unwrap();
        let _ = t1[&[0, 3]]; // Should panic - last dimension is 3, max index is 2
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn index_out_of_bounds_first_dim() {
        let t1 = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], None).unwrap();
        let _ = t1[&[2, 0]]; // Should panic - first dimension is 2, max index is 1
    }
}
