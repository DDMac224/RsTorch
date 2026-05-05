use rstorch::tensor::{Element, Tensor};

pub fn to_vec<T: Element>(t: &Tensor<T>) -> Vec<T> {
    let shape = t.shape();
    let total: usize = shape.iter().product();
    let mut result = Vec::with_capacity(total);

    fn recurse<T: Element>(t: &Tensor<T>, dims: &[usize], idx: &mut Vec<usize>, out: &mut Vec<T>) {
        if dims.is_empty() {
            out.push(t.index(idx.as_slice()).item().unwrap());
            return;
        }
        for i in 0..dims[0] {
            idx.push(i);
            recurse(t, &dims[1..], idx, out);
            idx.pop();
        }
    }

    recurse(t, &shape, &mut Vec::new(), &mut result);
    result
}
