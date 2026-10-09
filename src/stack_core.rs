// The Hasher's stack of chaining values in safe Rust: the library includes
// this file in src/lib.rs, and tools/verify/tree/rust/stackcore includes it
// too, for Aeneas (tools/verify/tree/HasherProofs.lean has the algorithm).
// It uses Cv and DEPTH from its includer.

/// The parent of two chaining values, the stack's one use of the kernels.
pub trait Parent {
    fn parent(&self, left: &Cv, right: &Cv) -> Cv;
}

/// Chaining values bottom first: an array made at the first push (a message
/// of one chunk pushes none) and its length.
#[derive(Clone)]
pub struct CvStack {
    cvs: Option<[Cv; DEPTH]>,
    len: usize,
}

impl CvStack {
    pub fn new() -> CvStack {
        CvStack { cvs: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Value `i`, below the length.
    pub fn get(&self, i: usize) -> Cv {
        assert!(i < self.len, "a value below the stack's length");
        match &self.cvs {
            Some(cvs) => cvs[i],
            None => panic!("a stack with values has its array"),
        }
    }

    /// `cv` on top; the stack holds fewer than DEPTH values.
    pub fn push(&mut self, cv: Cv) {
        if self.cvs.is_none() {
            self.cvs = Some([[0; 32]; DEPTH]);
        }
        if let Some(cvs) = &mut self.cvs {
            cvs[self.len] = cv;
        }
        self.len += 1;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// While the stack holds more than `target` values, its top two become
    /// their parent (merge_cv_stack). Two are there to merge whenever
    /// `target` is at least 1, as the Hasher's are.
    pub fn merge<P: Parent>(&mut self, p: &P, target: usize) {
        while self.len > target {
            assert!(self.len >= 2, "two values to merge");
            if let Some(cvs) = &mut self.cvs {
                let parent = p.parent(&cvs[self.len - 2], &cvs[self.len - 1]);
                cvs[self.len - 2] = parent;
            }
            self.len -= 1;
        }
    }
}
