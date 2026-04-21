use itertools::Itertools;

use crate::tensor::{Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn cpu_elemwise(&self, rhs: &Self, op: fn(T, T) -> T) -> Self {
        todo!()
    }

    fn matmul_matricies(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape.len(),
            2,
            "Matrix matmul can only have two dimensions"
        );
        assert_eq!(
            rhs.shape().len(),
            2,
            "Matrix matmul can only have two dimensions"
        );
        assert_eq!(
            self.shape[1],
            rhs.shape()[0],
            "Columns of self and rows of rhs must match"
        );

        let new_dims = vec![self.shape[0], rhs.shape()[1]];

        let self_rows = self.data.iter().skip(self.offset).step_by(self.stride[1]);
        let rhs_cols = rhs.data.iter().skip(rhs.offset).step_by(rhs.stride[0]);

        let products = self_rows.zip(rhs_cols).map(|(&s, &r)| s * r);
        let new_data = products
            .chunks(self.shape[1])
            .into_iter()
            .map(|chunk| chunk.reduce(|acc, e| acc + e).unwrap())
            .collect();

        Tensor::new(new_data, new_dims, Some(self.device()))
    }
}
