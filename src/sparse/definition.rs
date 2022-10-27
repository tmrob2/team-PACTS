use pyo3::prelude::*;

// TODO move the definition of a sparse matrix here#[repr(C)]
#[pyclass]
#[derive(Clone)]
pub struct CxxMatrixf32 {
    #[pyo3(get)]
    pub nzmax: i32,
    #[pyo3(get)]
    pub m: i32, // number of rows
    #[pyo3(get)]
    pub n: i32, // number of cols
    #[pyo3(get)]
    pub p: Vec<i32>,
    #[pyo3(get)]
    pub i: Vec<i32>,
    #[pyo3(get)]
    pub x: Vec<f32>,
    #[pyo3(get)]
    pub nz: i32, //non zero entries
}

#[repr(C)]
pub struct CSparsef32 {
    pub nz: i32,
    pub m: i32,
    pub n: i32,
    pub p: *const i32,
    pub i: *const i32,
    pub x: *const f32
} 

impl CxxMatrixf32 {
    pub fn make(nzmax: i32, m: i32, n: i32, nz: i32) -> Self {
        // makes a new triple
        CxxMatrixf32 { 
            nzmax, 
            m, 
            n, 
            p: Vec::new(), 
            i: Vec::new(), 
            x: Vec::new(), 
            nz
        }
    }

    pub fn new() -> Self {
        CxxMatrixf32 { 
            nzmax: 0, 
            m: 0, 
            n: 0, 
            p: Vec::new(), 
            i: Vec::new(), 
            x: Vec::new(), 
            nz: 0
        }
    }

    pub fn triple_entry(&mut self, i: i32, j: i32, val: f32) {
        self.i.push(i);
        self.p.push(j);
        self.x.push(val);
        self.nz += 1;
    }
}