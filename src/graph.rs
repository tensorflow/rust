use super::buffer::Buffer;
use super::AnyTensor;
use super::Code;
use super::DataType;
use super::Result;
use super::Shape;
use super::Status;
use super::Tensor;
use super::TensorType;
use libc::c_char;
use libc::c_float;
use libc::c_int;
use libc::c_uchar;
use libc::c_uint;
use libc::c_void;
use libc::size_t;
use std;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::os::raw::c_void as std_c_void;
use std::ptr;
use std::slice;
use std::str::Utf8Error;
use std::sync::Arc;
use tensorflow_sys as tf;

#[derive(Debug)]
struct GraphLifetime;

#[derive(Debug)]
struct GraphImpl {
    inner: *mut tf::TF_Graph,
    owned: bool,
}

unsafe impl Send for GraphImpl {}
unsafe impl Sync for GraphImpl {}

impl Drop for GraphImpl {
    /// Graph will be deleted once no more Sessions are referencing it.
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                tf::TF_DeleteGraph(self.inner);
            }
        }
    }
}

////////////////////////

/// `ImportGraphDefOptions` holds options that can be passed to
/// `Graph::import_graph_def`.
#[derive(Debug)]
pub struct ImportGraphDefOptions {
    inner: *mut tf::TF_ImportGraphDefOptions,
}

impl_new!(
    ImportGraphDefOptions,
    TF_NewImportGraphDefOptions,
    "Creates a default ImportGraphDefOptions."
);
impl_drop!(ImportGraphDefOptions, TF_DeleteImportGraphDefOptions);

impl ImportGraphDefOptions {
    /// Set the prefix to be prepended to the names of nodes in `graph_def` that will
    /// be imported into `graph`.
    pub fn set_prefix(&mut self, prefix: &str) -> std::result::Result<(), NulError> {
        let s = CString::new(prefix)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsSetPrefix(self.inner, s.as_ptr());
        }
        Ok(())
    }

    /// Set any imported nodes with input `src_name:src_index` to have that input
    /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
    /// `dst` references a node already existing in the graph being imported into.
    pub fn add_input_mapping(
        &mut self,
        src_name: &str,
        src_index: usize,
        dst: &Output,
    ) -> std::result::Result<(), NulError> {
        let s = CString::new(src_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsAddInputMapping(
                self.inner,
                s.as_ptr(),
                src_index as c_int,
                dst.to_c(),
            );
        }
        Ok(())
    }

    /// Set any imported nodes with control input `src_name` to have that input
    /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
    /// `dst` references an operation already existing in the graph being imported
    /// into.
    pub fn remap_control_dependency(
        &mut self,
        src_name: &str,
        dst: &Operation,
    ) -> std::result::Result<(), NulError> {
        let s = CString::new(src_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsRemapControlDependency(self.inner, s.as_ptr(), dst.inner);
        }
        Ok(())
    }

    /// Cause the imported graph to have a control dependency on `oper`. `oper`
    /// should exist in the graph being imported into.
    pub fn add_control_dependency(&mut self, oper: &Operation) {
        unsafe {
            tf::TF_ImportGraphDefOptionsAddControlDependency(self.inner, oper.inner);
        }
    }

    /// Add an output in `graph_def` to be returned via the `return_outputs` output
    /// parameter of `import_graph_def()`. If the output is remapped via an input
    /// mapping, the corresponding existing tensor in `graph` will be returned.
    pub fn add_return_output(
        &mut self,
        oper_name: &str,
        index: usize,
    ) -> std::result::Result<(), NulError> {
        let s = CString::new(oper_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsAddReturnOutput(self.inner, s.as_ptr(), index as c_int);
        }
        Ok(())
    }

    /// Add an operation in `graph_def` to be returned via the `return_opers` output
    /// parameter of import_graph_def().
    pub fn add_return_operation(&mut self, oper_name: &str) -> std::result::Result<(), NulError> {
        let s = CString::new(oper_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsAddReturnOperation(self.inner, s.as_ptr());
        }
        Ok(())
    }

    /// Returns the number of return outputs added via `add_return_output()`.
    pub fn num_return_outputs(&self) -> usize {
        unsafe { tf::TF_ImportGraphDefOptionsNumReturnOutputs(self.inner) as usize }
    }

    /// Returns the number of return operations added via `add_return_operation()`.
    pub fn num_return_operations(&self) -> usize {
        unsafe { tf::TF_ImportGraphDefOptionsNumReturnOperations(self.inner) as usize }
    }

    /// Set whether to uniquify imported operation names. If true, imported operation
    /// names will be modified if their name already exists in the graph. If false,
    /// conflicting names will be treated as an error. Note that this option has no
    /// effect if a prefix is set, since the prefix will guarantee all names are
    /// unique. Defaults to false.
    pub fn set_uniquify_names(&mut self, uniquify_names: bool) {
        unsafe {
            tf::TF_ImportGraphDefOptionsSetUniquifyNames(
                self.inner,
                if uniquify_names { 1 } else { 0 },
            );
        }
    }

    /// If true, the specified prefix will be modified if it already exists as an
    /// operation name or prefix in the graph. If false, a conflicting prefix will be
    /// treated as an error. This option has no effect if no prefix is specified.
    pub fn set_uniquify_prefix(&mut self, uniquify_prefix: bool) {
        unsafe {
            tf::TF_ImportGraphDefOptionsSetUniquifyPrefix(
                self.inner,
                if uniquify_prefix { 1 } else { 0 },
            );
        }
    }

    /// Set the execution device for nodes.
    /// Only applies to nodes where a device was not already explicitly specified.
    pub fn set_default_device(&mut self, device: &str) -> std::result::Result<(), NulError> {
        let s = CString::new(device)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsSetDefaultDevice(self.inner, s.as_ptr());
        }
        Ok(())
    }
}

////////////////////////

/// ImportGraphDefResults holds results that are generated by
/// Graph::import_graph_def_with_results().
#[derive(Debug)]
pub struct ImportGraphDefResults {
    inner: *mut tf::TF_ImportGraphDefResults,
    gimpl: Arc<GraphImpl>,
}

impl ImportGraphDefResults {
    /// Fetches the return outputs requested via ImportGraphDefOptions::add_return_output().
    pub fn return_outputs(&self) -> Vec<Output> {
        unsafe {
            let mut num_outputs: c_int = 0;
            let mut c_outputs: *mut tf::TF_Output = ptr::null_mut();
            tf::TF_ImportGraphDefResultsReturnOutputs(self.inner, &mut num_outputs, &mut c_outputs);
            slice::from_raw_parts(c_outputs, num_outputs as usize)
                .iter()
                .map(|output| Output {
                    operation: Operation {
                        inner: output.oper,
                        gimpl: self.gimpl.clone(),
                    },
                    index: output.index,
                })
                .collect()
        }
    }

    /// Fetches the return operations requested via ImportGraphDefOptions::add_return_operation().
    pub fn return_operations(&self) -> Vec<Operation> {
        unsafe {
            let mut num_operations: c_int = 0;
            let mut c_operations: *mut *mut tf::TF_Operation = ptr::null_mut();
            tf::TF_ImportGraphDefResultsReturnOperations(
                self.inner,
                &mut num_operations,
                &mut c_operations,
            );
            slice::from_raw_parts(c_operations, num_operations as usize)
                .iter()
                .map(|operation| Operation {
                    inner: *operation,
                    gimpl: self.gimpl.clone(),
                })
                .collect()
        }
    }

    /// Fetches any input mappings requested via
    /// ImportGraphDefOptions::add_input_mapping() that didn't appear in the GraphDef
    /// and weren't used as input to any node in the imported graph def.
    pub fn missing_unused_input_mappings(
        &self,
    ) -> std::result::Result<Vec<(&str, c_int)>, Utf8Error> {
        unsafe {
            let mut n: c_int = 0;
            let mut c_src_names: *mut *const c_char = ptr::null_mut();
            let mut src_indexes: *mut c_int = ptr::null_mut();
            tf::TF_ImportGraphDefResultsMissingUnusedInputMappings(
                self.inner,
                &mut n,
                &mut c_src_names,
                &mut src_indexes,
            );
            let c_name_slice = slice::from_raw_parts(c_src_names, n as usize);
            let index_slice = slice::from_raw_parts(src_indexes, n as usize);
            let mut v = Vec::new();
            for i in 0..n as usize {
                let s = CStr::from_ptr(c_name_slice[i]).to_str()?;
                v.push((s, index_slice[i]));
            }
            Ok(v)
        }
    }
}

impl_drop!(ImportGraphDefResults, TF_DeleteImportGraphDefResults);

////////////////////////

/// Represents a computation graph.  Graphs may be shared between sessions.
/// Graphs are thread-safe when used as directed.
#[derive(Debug)]
pub struct Graph {
    gimpl: Arc<GraphImpl>,
    lifetime: GraphLifetime,
}

impl Graph {
    /// Creates a new graph.
    pub fn new() -> Graph {
        unsafe {
            Graph {
                gimpl: Arc::new(GraphImpl {
                    inner: tf::TF_NewGraph(),
                    owned: true,
                }),
                lifetime: GraphLifetime,
            }
        }
    }

    /// Operation will only be added to graph when finish_operation() is called
    /// (assuming finish_operation() does not return an error).  graph must
    /// not be deleted until after finish_operation() is called.
    pub fn new_operation(
        &mut self,
        op_type: &str,
        operation_name: &str,
    ) -> std::result::Result<OperationDescription<'_>, NulError> {
        let c_op_type = CString::new(op_type)?;
        let c_operation_name = CString::new(operation_name)?;
        unsafe {
            Ok(OperationDescription {
                inner: tf::TF_NewOperation(
                    self.gimpl.inner,
                    c_op_type.as_ptr(),
                    c_operation_name.as_ptr(),
                ),
                graph: self,
                finished: false,
            })
        }
    }

    /// Returns the operation in the graph with the given name, if it exists.
    /// If the operation does not exist, returns `Ok(None)`.
    pub fn operation_by_name(
        &self,
        operation_name: &str,
    ) -> std::result::Result<Option<Operation>, NulError> {
        let c_operation_name = CString::new(operation_name)?;
        unsafe {
            let operation =
                tf::TF_GraphOperationByName(self.gimpl.inner, c_operation_name.as_ptr());
            if operation.is_null() {
                Ok(None)
            } else {
                Ok(Some(Operation {
                    inner: operation,
                    gimpl: self.gimpl.clone(),
                }))
            }
        }
    }

    /// Like `operation_by_name`, except that failure to find the operation is considered an error.
    pub fn operation_by_name_required(
        &self,
        operation_name: &str,
    ) -> std::result::Result<Operation, Status> {
        match self.operation_by_name(operation_name)? {
            Some(operation) => Ok(operation),
            None => Err(Status::new_set(
                Code::Unavailable,
                &format!("Operation {:?} not found", operation_name),
            )
            .unwrap()),
        }
    }

    /// Finds a unique operation name.  The pattern must contain exactly one
    /// '{}' placeholder to indicate where a unique ID can be inserted, e.g.
    /// 'Add_{}' or 'while_loop_{}/Merge', and the function returns an integer
    /// which, when inserted into the placeholder, yields an operation name
    /// which does not appear in the graph.
    pub(crate) fn generate_operation_name(&self, operation_name_pattern: &str) -> Result<i64> {
        let parts: Vec<_> = operation_name_pattern.split("{}").collect();
        if parts.len() != 2 {
            return Err(invalid_arg!(
                "operation_name_pattern must contain placeholder"
            ));
        }
        // Can't use format! because its argument must be a string literal.
        let mut i = 0;
        loop {
            let name = format!("{}{}{}", parts[0], i, parts[1]);
            let c_name = CString::new(name)?;
            unsafe {
                if tf::TF_GraphOperationByName(self.gimpl.inner, c_name.as_ptr()).is_null() {
                    return Ok(i);
                }
            }
            i += 1;
        }
    }

    /// Iterates over the operations in the graph.
    pub fn operation_iter(&self) -> OperationIter<'_> {
        OperationIter {
            graph: self,
            pos: 0,
        }
    }

    /// Returns the graph definition as a protobuf.
    pub fn graph_def(&self) -> Result<Vec<u8>> {
        let mut status = Status::new();
        unsafe {
            let c_buffer = tf::TF_NewBuffer();
            tf::TF_GraphToGraphDef(self.gimpl.inner, c_buffer, status.inner());
            if status.is_ok() {
                Ok(Buffer::from_c(c_buffer, true).into())
            } else {
                tf::TF_DeleteBuffer(c_buffer);
                Err(status)
            }
        }
    }

    /// Returns the number of dimensions of the Tensor referenced by `output`.
    ///
    /// If the number of dimensions in the shape is unknown, returns -1.
    ///
    /// Returns an error if:
    ///
    ///   * `output` is not in `graph`.
    pub fn num_dims<I: Into<Output>>(&self, output: I) -> Result<c_int> {
        let mut status = Status::new();
        unsafe {
            let val = tf::TF_GraphGetTensorNumDims(
                self.gimpl.inner,
                output.into().to_c(),
                status.inner(),
            );
            if status.is_ok() {
                Ok(val)
            } else {
                Err(status)
            }
        }
    }

    /// Returns the shape of the Tensor referenced by `output`.
    ///
    /// Returns an error if:
    ///
    ///   * `output` is not in `graph`.
    pub fn tensor_shape<I: Into<Output>>(&self, output: I) -> Result<Shape> {
        let mut status = Status::new();
        let output = output.into();
        let n = self.num_dims(output.clone())?;
        if n == -1 {
            return Ok(Shape(None));
        }
        let mut dims = Vec::with_capacity(n as usize);
        unsafe {
            tf::TF_GraphGetTensorShape(
                self.gimpl.inner,
                output.to_c(),
                dims.as_mut_ptr(),
                n,
                status.inner(),
            );
            if status.is_ok() {
                dims.set_len(n as usize);
                Ok(Shape(Some(
                    dims.iter()
                        .map(|x| if *x < 0 { None } else { Some(*x) })
                        .collect(),
                )))
            } else {
                Err(status)
            }
        }
    }

    /// Import the graph serialized in `graph_def`.
    pub fn import_graph_def(
        &mut self,
        graph_def: &[u8],
        options: &ImportGraphDefOptions,
    ) -> Result<()> {
        let buf = Buffer::from(graph_def);
        let mut status = Status::new();
        unsafe {
            tf::TF_GraphImportGraphDef(
                self.gimpl.inner,
                buf.inner(),
                options.inner,
                status.inner(),
            );
            status.into_result()
        }
    }

    /// Import the graph serialized in `graph_def`.
    pub fn import_graph_def_with_results(
        &mut self,
        graph_def: &[u8],
        options: &ImportGraphDefOptions,
    ) -> Result<ImportGraphDefResults> {
        let buf = Buffer::from(graph_def);
        let mut status = Status::new();
        unsafe {
            let result = tf::TF_GraphImportGraphDefWithResults(
                self.gimpl.inner,
                buf.inner(),
                options.inner,
                status.inner(),
            );
            status.into_result().map(|()| ImportGraphDefResults {
                inner: result,
                gimpl: self.gimpl.clone(),
            })
        }
    }

    /// Import the graph serialized in `graph_def`.
    pub fn import_graph_def_with_return_outputs(
        &mut self,
        graph_def: &[u8],
        options: &ImportGraphDefOptions,
    ) -> Result<Vec<Output>> {
        let buf = Buffer::from(graph_def);
        let mut status = Status::new();
        let n = options.num_return_outputs();
        let mut c_return_outputs = Vec::with_capacity(n);
        unsafe {
            c_return_outputs.set_len(n);
            tf::TF_GraphImportGraphDefWithReturnOutputs(
                self.gimpl.inner,
                buf.inner(),
                options.inner,
                c_return_outputs.as_mut_ptr(),
                n as c_int,
                status.inner(),
            );
        }
        status.into_result()?;
        Ok(c_return_outputs
            .iter()
            .map(|x| Output::from_c(self, x))
            .collect())
    }

    /// Adds a copy of function `func` and optionally its gradient function
    /// `grad` to the graph. Once `func`/`grad` is added to the graph, it can be
    /// called by creating an operation using the function's name. Any changes
    /// to `func`/`grad` (including deleting it) done after this method returns,
    /// won't affect the copy of `func`/`grad` in the graph. If `func` or `grad`
    /// are already in the graph, `copy_function` has no effect on them, but can
    /// establish the function->gradient relationship between them if `func`
    /// does not already have a gradient. If `func` already has a gradient
    /// different from `grad`, an error is returned.
    ///
    /// If `grad` is None and `func` is not in the graph, `func` is added
    /// without a gradient. If `grad` is None and `func` is in the graph,
    /// `copy_function` is a noop. `grad` must have appropriate signature as
    /// described in the doc of GradientDef in
    /// tensorflow/core/framework/function.proto.
    ///
    /// If successful, returns () and `func` and `grad` are added to the graph.
    /// Otherwise, an error is returned and the graph is unmodified.
    pub fn copy_function(&mut self, func: &Function, grad: Option<&Function>) -> Result<()> {
        let mut status = Status::new();
        unsafe {
            tf::TF_GraphCopyFunction(
                self.inner(),
                func.inner,
                match grad {
                    None => ptr::null(),
                    Some(g) => g.inner,
                },
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Create a `Function` from a `Graph`.
    ///
    /// # Arguments
    ///
    /// * `fn_name` - the name of the new `Function`. Should match the operation
    ///   name (OpDef.name) regexp [A-Z][A-Za-z0-9_.\\-/]*. If
    ///   `append_hash_to_fn_name` is false, `fn_name` must be distinct from
    ///   other function and operation names (at least those registered in
    ///   graphs where this function will be used).
    /// * `append_hash_to_fn_name` - If true, the actual name of the function
    ///   will be `fn_name` appended with
    ///   '_&lt;hash_of_this_function's_definition&gt;'. If false, the
    ///   function's name will be `fn_name`.
    /// * `opers` - Array of operations to become the body of the function or
    ///   null.
    ///   * If `None`, all the operations in the graph will become part of the
    ///     function except operations referenced in `inputs`. These operations
    ///     must have a single output (these operations are typically
    ///     placeholders created for the sole purpose of representing an input.
    ///     We can relax this constraint if there are compelling use cases).
    ///   * If `Some`, all operations in it will become part of the function. In
    ///     particular, no automatic skipping of dummy input operations is
    ///     performed.
    /// * `inputs` - array of `Output`s that specify the inputs to the function.
    ///   The names used for function inputs are normalized names of the
    ///   operations (usually placeholders) pointed to by `inputs`. These
    ///   operation names should start with a letter. Normalization will convert
    ///   all letters to lowercase and non-alphanumeric characters to '\_' to
    ///   make resulting names match the "[a-z][a-z0-9_]*" pattern for operation
    ///   argument names. `inputs` cannot contain the same tensor twice.
    /// * `outputs` - array of `Output`s that specify the outputs of the
    ///   function. `outputs` can contain the same tensor more than once.
    /// * `output_names` - The names of the function's outputs. `output_names`
    ///   array must either have the same length as `outputs` or be None. In the
    ///   former case, the names should match the regular expression for ArgDef
    ///   names - "[a-z][a-z0-9_]*". In the latter case, names for outputs will
    ///   be generated automatically.
    /// * `opts` - various options for the function, e.g. XLA's inlining control.
    /// * `description` - optional human-readable description of this function.
    ///
    /// Note that when the same `Output` is listed as both an input and an
    /// output, the corresponding function's output will equal to this input,
    /// instead of the original node's output.
    ///
    /// Callers must also satisfy the following constraints:
    ///
    /// * `inputs` cannot refer to `Output`s within a control flow context. For
    ///   example, one cannot use the output of "switch" node as input.
    /// * `inputs` and `outputs` cannot have reference types. Reference types
    ///   are not exposed through C API and are being replaced with Resources.
    ///   We support reference types inside function's body to support legacy
    ///   code. Do not use them in new code.
    /// * Every node in the function's body must have all of its inputs
    ///   (including control inputs). In other words, for every node in the
    ///   body, each input must be either listed in `inputs` or must come from
    ///   another node in the body. In particular, it is an error to have a
    ///   control edge going from a node outside of the body into a node in the
    ///   body. This applies to control edges going from nodes referenced in
    ///   `inputs` to nodes in the body when the former nodes are not in the
    ///   body (automatically skipped or not included in explicitly specified
    ///   body).
    ///
    /// # Returns
    ///
    ///  A newly created `Function` instance.
    pub fn to_function<S: AsRef<str>>(
        &self,
        fn_name: &str,
        append_hash_to_fn_name: bool,
        opers: Option<&[&Operation]>,
        inputs: &[Output],
        outputs: &[Output],
        output_names: Option<&[S]>,
        opts: &FunctionOptions,
        description: Option<&str>,
    ) -> Result<Function> {
        let fn_name_cstr = CString::new(fn_name)?;
        let num_opers: c_int = if let &Some(ops) = &opers {
            ops.len() as c_int
        } else {
            -1
        };
        #[allow(trivial_casts)]
        let c_opers: Option<Vec<_>> =
            opers.map(|s| s.iter().map(|op| op.inner as *const _).collect());
        let c_opers_ptr: *const *const tf::TF_Operation = if let &Some(ref ops) = &c_opers {
            ops.as_ptr()
        } else {
            ptr::null()
        };
        let c_inputs: Vec<_> = inputs.iter().map(|x| x.to_c()).collect();
        let c_outputs: Vec<_> = outputs.iter().map(|x| x.to_c()).collect();
        let output_names_cstrs: Option<::std::result::Result<Vec<CString>, NulError>> =
            output_names
                .map(|slice: &[S]| slice.iter().map(|s: &S| CString::new(s.as_ref())).collect());
        let output_names_cstrs: Option<Vec<CString>> = match output_names_cstrs {
            None => None,
            Some(r) => Some(r?),
        };
        // Don't use Option::map because the CStrings need to outlive the
        // pointers and Option::map consumes the Option.
        let output_names_ptrs: Option<Vec<*const c_char>> = match &output_names_cstrs {
            &None => None,
            &Some(ref slice) => Some(slice.iter().map(|s| s.as_ptr()).collect()),
        };
        let output_names_ptrs_ptr = match &output_names_ptrs {
            &None => ptr::null(),
            &Some(ref v) => v.as_ptr(),
        };
        let description_cstr = match description {
            None => None,
            Some(d) => Some(CString::new(d)?),
        };
        let description_ptr: *const c_char = if let &Some(ref cstr) = &description_cstr {
            cstr.as_ptr()
        } else {
            ptr::null()
        };
        let status = Status::new();
        let f = unsafe {
            tf::TF_GraphToFunction(
                self.inner(),
                fn_name_cstr.as_ptr(),
                if append_hash_to_fn_name { 1 } else { 0 },
                num_opers,
                c_opers_ptr,
                c_inputs.len() as c_int,
                c_inputs.as_ptr(),
                c_outputs.len() as c_int,
                c_outputs.as_ptr(),
                output_names_ptrs_ptr,
                opts.inner,
                description_ptr,
                status.inner,
            )
        };
        status.into_result()?;
        Ok(Function { inner: f })
    }

    /// Returns the number of functions registered in the graph.
    pub fn num_functions(&self) -> c_int {
        unsafe { tf::TF_GraphNumFunctions(self.inner()) }
    }

    /// Returns functions registered in the graph.
    pub fn get_functions(&self) -> Result<Vec<Function>> {
        unsafe {
            let num = tf::TF_GraphNumFunctions(self.inner());
            let mut funcs = Vec::with_capacity(num as usize);
            let status = Status::new();
            let num = tf::TF_GraphGetFunctions(self.inner(), funcs.as_mut_ptr(), num, status.inner);
            status.into_result()?;
            funcs.set_len(num as usize);
            Ok(funcs.iter().map(|f| Function { inner: *f }).collect())
        }
    }

    /// Returns the serialized OpDef proto with name `op_name`, or a bad status if no
    /// such op exists. This can return OpDefs of functions copied into the graph.
    pub fn get_op_def(&self, op_name: &str) -> Result<Vec<u8>> {
        let status = Status::new();
        let c_op_name = CString::new(op_name)?;
        unsafe {
            let mut buffer = Buffer::new_unallocated();
            tf::TF_GraphGetOpDef(
                self.inner(),
                c_op_name.as_ptr(),
                buffer.inner_mut(),
                status.inner,
            );
            status.into_result().map(|()| buffer.into())
        }
    }

    /// Returns the serialized VersionDef proto for this graph.
    pub fn versions(&self) -> Result<Vec<u8>> {
        let status = Status::new();
        unsafe {
            let mut buffer = Buffer::new_unallocated();
            tf::TF_GraphVersions(self.inner(), buffer.inner_mut(), status.inner);
            status.into_result().map(|()| buffer.into())
        }
    }

    /// Attempts to evaluate `output`. This will only be possible if `output`
    /// doesn't depend on any graph inputs (this function is safe to call if
    /// this isn't the case though).
    ///
    /// If the evaluation is successful, this function returns the tensor.
    /// Otherwise returns None. An error status is returned if something is
    /// wrong with the graph or input or the type requested doesn't match the
    /// type of the tensor.
    pub fn try_evaluate_constant<T: TensorType>(
        &self,
        output: &Output,
    ) -> Result<Option<Tensor<T>>> {
        let status = Status::new();
        unsafe {
            let mut c_tensor: *mut tf::TF_Tensor = ptr::null_mut();
            let success = tf::TF_TryEvaluateConstant(
                self.inner(),
                output.to_c(),
                &mut c_tensor,
                status.inner,
            );
            status.into_result()?;
            if success != 0 {
                match Tensor::from_tf_tensor(c_tensor) {
                    None => Err(invalid_arg!("Tensor types do not match")),
                    Some(t) => Ok(Some(t)),
                }
            } else {
                Ok(None)
            }
        }
    }

    /// Adds operations to compute the partial derivatives of sum of `y`s
    /// w.r.t `x`s, i.e., d(y_1 + y_2 + ...)/dx_1, d(y_1 + y_2 + ...)/dx_2...
    ///
    /// `dx` are used as initial gradients (which represent the symbolic partial
    /// derivatives of some loss function `L` w.r.t. `y`).
    /// `dx` must be None or have the same length as `y`.
    /// If `dx` is None, the implementation will use dx of `OnesLike` for all
    /// shapes in `y`.
    /// `prefix` names the scope into which all gradients operations are being
    /// added.  `prefix` must be unique within the provided graph otherwise this
    /// operation will fail. If `prefix` is None, gradient nodes are
    /// automatically named under the "gradients/" prefix. To guarantee name
    /// uniqueness, subsequent calls to the same graph will append an
    /// incremental tag to the prefix: "gradients_1/", "gradients_2/", ...
    ///
    /// WARNING: This function does not yet support all the gradients that
    /// python supports. See
    /// https://www.tensorflow.org/code/tensorflow/cc/gradients/README.md
    /// for instructions on how to add C++ more gradients.
    pub fn add_gradients(
        &mut self,
        prefix: Option<&str>,
        y: &[Output],
        x: &[Output],
        dx: Option<&[Output]>,
    ) -> Result<Vec<Option<Output>>> {
        if let Some(dx) = dx {
            if dx.len() != y.len() {
                return Err(invalid_arg!(
                    "dx.len() must equal y.len() ({} vs. {})",
                    dx.len(),
                    y.len()
                ));
            }
        }
        let c_y: Vec<_> = y.iter().map(Output::to_c).collect();
        let c_x: Vec<_> = x.iter().map(Output::to_c).collect();
        let c_dx: Option<Vec<_>> = dx.map(|v| v.iter().map(Output::to_c).collect());
        let dx_ptr = match c_dx {
            Some(v) => v.as_ptr(),
            None => ptr::null(),
        };
        let prefix_cstr = match prefix {
            Some(s) => Some(CString::new(s)?),
            None => None,
        };
        let prefix_ptr: *const c_char = if let &Some(ref cstr) = &prefix_cstr {
            cstr.as_ptr()
        } else {
            ptr::null()
        };
        let mut dy = Vec::with_capacity(x.len());
        let mut status = Status::new();
        unsafe {
            tf::TF_AddGradientsWithPrefix(
                self.inner(),
                prefix_ptr,
                c_y.as_ptr() as *mut _,
                y.len() as i32,
                c_x.as_ptr() as *mut _,
                x.len() as i32,
                dx_ptr as *mut _,
                status.inner(),
                dy.as_mut_ptr(),
            );
            if status.is_ok() {
                dy.set_len(x.len());
                Ok(dy
                    .iter()
                    .map(|o| Output::from_c_optional(self, o))
                    .collect())
            } else {
                Err(status)
            }
        }
    }

    pub(crate) fn inner(&self) -> *mut tf::TF_Graph {
        self.gimpl.inner
    }

    pub(crate) unsafe fn from_c(inner: *mut tf::TF_Graph) -> Self {
        Graph {
            gimpl: Arc::new(GraphImpl {
                inner,
                owned: false,
            }),
            lifetime: GraphLifetime,
        }
    }
}

////////////////////////

/// Iterator over the operations in a `Graph`.
#[derive(Debug)]
pub struct OperationIter<'a> {
    // We could just have a gimpl field, but keeping a reference to the Graph
    // means that the graph can't be modified while iterating through it.
    graph: &'a Graph,
    pos: size_t,
}

impl<'a> Iterator for OperationIter<'a> {
    type Item = Operation;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let operation = tf::TF_GraphNextOperation(self.graph.gimpl.inner, &mut self.pos);
            if operation.is_null() {
                None
            } else {
                Some(Operation {
                    inner: operation,
                    gimpl: self.graph.gimpl.clone(),
                })
            }
        }
    }
}

////////////////////////

c_enum!(
    TF_AttrType,
    // TODO: Provide docs on variants once they are added to c_api.h.
    /// Describes the type of the value of an attribute on an operation.
    #[allow(missing_docs)]
    AttrType {
        String = 0,
        Int = 1,
        Float = 2,
        Bool = 3,
        Type = 4,
        Shape = 5,
        Tensor = 6,
        Placeholder = 7,
        Func = 8,
    });

/// AttrMetadata describes the value of an attribute on an operation.
#[derive(Clone, Debug, Copy)]
pub struct AttrMetadata {
    /// Length of the list, or None if the attribute is not a list.
    pub list_size: Option<i64>,

    /// Type of elements of the list if the attribute is a list.
    /// Type of the single value stored in the attribute if not a list.
    pub attr_type: AttrType,

    /// Total size the attribute value.
    /// The units of total_size depend on list_size and attr_type.
    ///
    /// 1. If attr_type == AttrType::String and list_size == None
    ///    then total_size is the byte size of the string valued attribute.
    /// 2. If attr_type == AttrType::String and list_size == Some(_)
    ///    then total_size is the cumulative byte size of all the strings in the
    ///    list.
    /// 3. If attr_type == AttrType::Shape and list_size == None
    ///    then total_size is the number of dimensions of the shape valued
    ///    attribute, or -1 if its rank is unknown.
    /// 4. If attr_type == AttrType::SHAPE and list_size == Some(_)
    ///    then total_size is the cumulative number of dimensions of all shapes
    ///    in the list.
    /// 4. Otherwise, total_size is undefined.
    pub total_size: i64,
}

impl AttrMetadata {
    fn from_c(metadata: tf::TF_AttrMetadata) -> Self {
        AttrMetadata {
            list_size: if metadata.is_list == 0 {
                None
            } else {
                Some(metadata.list_size)
            },
            attr_type: AttrType::from_c(metadata.type_),
            total_size: metadata.total_size,
        }
    }
}

////////////////////////

/// An `Operation` is a node in a `Graph`.
/// It is a computation which accepts inputs and produces outputs.
#[derive(Debug, Clone)]
pub struct Operation {
    inner: *mut tf::TF_Operation,
    gimpl: Arc<GraphImpl>,
}

unsafe impl Send for Operation {}
unsafe impl Sync for Operation {}

impl Operation {
    /// Returns the name of the operation.
    ///
    /// This is the name of the specific computational step,
    /// not an operation type, so it may look like `'add_x_and_y'` instead of `'Add'`,
    /// although it may be a generated ID like `'Add_123'`.
    pub fn name(&self) -> std::result::Result<String, Utf8Error> {
        unsafe {
            CStr::from_ptr(tf::TF_OperationName(self.inner))
                .to_str()
                .map(|x| x.to_string())
        }
    }

    /// Returns the type of operation.
    /// This will be something like `'Add'`, `'Mul'`, etc.
    pub fn op_type(&self) -> std::result::Result<String, Utf8Error> {
        unsafe {
            CStr::from_ptr(tf::TF_OperationOpType(self.inner))
                .to_str()
                .map(|x| x.to_string())
        }
    }

    /// Returns the device for this operation.
    /// The empty string means unconstrained.
    pub fn device(&self) -> std::result::Result<String, Utf8Error> {
        unsafe {
            CStr::from_ptr(tf::TF_OperationDevice(self.inner))
                .to_str()
                .map(|x| x.to_string())
        }
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        unsafe { tf::TF_OperationNumOutputs(self.inner) as usize }
    }

    /// Returns the type of a specific output.
    pub fn output_type(&self, index: usize) -> DataType {
        unsafe {
            DataType::from_c(tf::TF_OperationOutputType(tf::TF_Output {
                oper: self.inner,
                index: index as c_int,
            }))
        }
    }

    // TODO: Figure out what this does and document it.
    #[allow(missing_docs)]
    pub fn output_list_length(&self, arg_name: &str) -> Result<usize> {
        let c_arg_name = CString::new(arg_name)?;
        let mut status = Status::new();
        let length = unsafe {
            tf::TF_OperationOutputListLength(self.inner, c_arg_name.as_ptr(), status.inner())
        };
        if status.is_ok() {
            Ok(length as usize)
        } else {
            Err(status)
        }
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        unsafe { tf::TF_OperationNumInputs(self.inner) as usize }
    }

    /// Returns the type of a specific input.
    pub fn input_type(&self, index: usize) -> DataType {
        unsafe {
            DataType::from_c(tf::TF_OperationInputType(tf::TF_Input {
                oper: self.inner,
                index: index as c_int,
            }))
        }
    }

    // TODO: Figure out what this does and document it.
    #[allow(missing_docs)]
    pub fn input_list_length(&self, arg_name: &str) -> Result<usize> {
        let c_arg_name = CString::new(arg_name)?;
        let mut status = Status::new();
        let length = unsafe {
            tf::TF_OperationInputListLength(self.inner, c_arg_name.as_ptr(), status.inner())
        };
        if status.is_ok() {
            Ok(length as usize)
        } else {
            Err(status)
        }
    }

    /// Returns the given input edge.
    /// The index argument is the index into the current operation's input array,
    /// and the return value is the source operation and the index into its output array.
    pub fn input(&self, index: usize) -> (Operation, usize) {
        unsafe {
            let port = tf::TF_OperationInput(tf::TF_Input {
                oper: self.inner,
                index: index as c_int,
            });
            (
                Operation {
                    inner: port.oper,
                    gimpl: self.gimpl.clone(),
                },
                port.index as usize,
            )
        }
    }

    /// Returns the number of consumers of a specific output.
    pub fn output_num_consumers(&self, index: usize) -> usize {
        unsafe {
            tf::TF_OperationOutputNumConsumers(tf::TF_Output {
                oper: self.inner,
                index: index as c_int,
            }) as usize
        }
    }

    /// Returns the consumers of a specific output.
    /// The index argument is the index into the current operation's output array,
    /// and the return value is a vector of the destination operation and the index
    /// into its input array.
    pub fn output_consumers(&self, index: usize) -> Vec<(Operation, usize)> {
        unsafe {
            let num_consumers = tf::TF_OperationOutputNumConsumers(tf::TF_Output {
                oper: self.inner,
                index: index as c_int,
            });
            let mut vec = <Vec<tf::TF_Input>>::with_capacity(num_consumers as usize);
            let len = tf::TF_OperationOutputConsumers(
                tf::TF_Output {
                    oper: self.inner,
                    index: index as c_int,
                },
                vec.as_mut_ptr(),
                vec.len() as c_int,
            );
            vec.set_len(len as usize);
            vec.into_iter()
                .map(|port| {
                    (
                        Operation {
                            inner: port.oper,
                            gimpl: self.gimpl.clone(),
                        },
                        port.index as usize,
                    )
                })
                .collect()
        }
    }

    /// Returns the number of control inputs.
    pub fn num_control_inputs(&self) -> usize {
        unsafe { tf::TF_OperationNumControlInputs(self.inner) as usize }
    }

    /// Returns the control inputs.
    pub fn control_inputs(&self) -> Vec<Operation> {
        unsafe {
            let num_consumers = tf::TF_OperationNumControlInputs(self.inner);
            let mut vec = <Vec<*mut tf::TF_Operation>>::with_capacity(num_consumers as usize);
            let len =
                tf::TF_OperationGetControlInputs(self.inner, vec.as_mut_ptr(), vec.len() as c_int);
            vec.set_len(len as usize);
            vec.into_iter()
                .map(|operation| Operation {
                    inner: operation,
                    gimpl: self.gimpl.clone(),
                })
                .collect()
        }
    }

    /// Returns the number of control outputs.
    pub fn num_control_outputs(&self) -> usize {
        unsafe { tf::TF_OperationNumControlOutputs(self.inner) as usize }
    }

    /// Returns the control outputs.
    pub fn control_outputs(&self) -> Vec<Operation> {
        unsafe {
            let num_consumers = tf::TF_OperationNumControlOutputs(self.inner);
            let mut vec = <Vec<*mut tf::TF_Operation>>::with_capacity(num_consumers as usize);
            let len =
                tf::TF_OperationGetControlOutputs(self.inner, vec.as_mut_ptr(), vec.len() as c_int);
            vec.set_len(len as usize);
            vec.into_iter()
                .map(|operation| Operation {
                    inner: operation,
                    gimpl: self.gimpl.clone(),
                })
                .collect()
        }
    }

    /// Returns metadata about the value of the attribute `attr_name`.
    pub fn get_attr_metadata(&self, attr_name: &str) -> Result<AttrMetadata> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if status.is_ok() {
                Ok(AttrMetadata::from_c(metadata))
            } else {
                Err(status)
            }
        }
    }

    /// Returns the value of the attribute `attr_name`.
    pub fn get_attr_string(&self, attr_name: &str) -> Result<String> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut v: Vec<u8> = Vec::with_capacity(metadata.total_size as usize);
            v.set_len(metadata.total_size as usize);
            tf::TF_OperationGetAttrString(
                self.inner,
                c_attr_name.as_ptr(),
                v.as_mut_ptr() as *mut std::os::raw::c_void,
                metadata.total_size as usize,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            Ok(CString::new(v)?.into_string()?)
        }
    }

    /// Get the list of strings in the value of the attribute `attr_name`.
    pub fn get_attr_string_list(&self, attr_name: &str) -> Result<Vec<String>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut storage: Vec<u8> = Vec::with_capacity(metadata.total_size as usize);
            storage.set_len(metadata.total_size as usize);
            let mut values: Vec<*const std::os::raw::c_char> =
                Vec::with_capacity(metadata.list_size as usize);
            let mut lengths: Vec<size_t> = Vec::with_capacity(metadata.list_size as usize);
            tf::TF_OperationGetAttrStringList(
                self.inner,
                c_attr_name.as_ptr(),
                values.as_mut_ptr() as *mut *mut std::os::raw::c_void,
                lengths.as_mut_ptr(),
                metadata.list_size as i32,
                storage.as_mut_ptr() as *mut std::os::raw::c_void,
                metadata.total_size as usize,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            values.set_len(metadata.list_size as usize);
            lengths.set_len(metadata.list_size as usize);
            let mut strings = Vec::with_capacity(metadata.list_size as usize);
            for i in 0..metadata.list_size as usize {
                let s = slice::from_raw_parts(values[i] as *const u8, lengths[i]);
                strings.push(std::str::from_utf8(s)?.to_string());
            }
            Ok(strings)
        }
    }

    /// Returns the value of the attribute `attr_name`.
    pub fn get_attr_int(&self, attr_name: &str) -> Result<i64> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        let mut value: i64 = 0;
        unsafe {
            tf::TF_OperationGetAttrInt(
                self.inner,
                c_attr_name.as_ptr(),
                &mut value,
                status.inner(),
            );
        }
        if !status.is_ok() {
            return Err(status);
        }
        Ok(value)
    }

    /// Get the list of ints in the value of the attribute `attr_name`.
    pub fn get_attr_int_list(&self, attr_name: &str) -> Result<Vec<i64>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut values: Vec<i64> = Vec::with_capacity(metadata.list_size as usize);
            values.set_len(metadata.list_size as usize);
            tf::TF_OperationGetAttrIntList(
                self.inner,
                c_attr_name.as_ptr(),
                values.as_mut_ptr(),
                metadata.list_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            Ok(values)
        }
    }

    /// Returns the value of the attribute `attr_name`.
    pub fn get_attr_float(&self, attr_name: &str) -> Result<f32> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        let mut value: c_float = 0.0;
        unsafe {
            tf::TF_OperationGetAttrFloat(
                self.inner,
                c_attr_name.as_ptr(),
                &mut value,
                status.inner(),
            );
        }
        if !status.is_ok() {
            return Err(status);
        }
        #[allow(trivial_numeric_casts)]
        Ok(value as f32)
    }

    /// Get the list of floats in the value of the attribute `attr_name`.
    pub fn get_attr_float_list(&self, attr_name: &str) -> Result<Vec<f32>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut values: Vec<c_float> = Vec::with_capacity(metadata.list_size as usize);
            values.set_len(metadata.list_size as usize);
            tf::TF_OperationGetAttrFloatList(
                self.inner,
                c_attr_name.as_ptr(),
                values.as_mut_ptr(),
                metadata.list_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            #[allow(trivial_numeric_casts)]
            Ok(values.iter().map(|f| *f as f32).collect())
        }
    }

    /// Returns the value of the attribute `attr_name`.
    pub fn get_attr_bool(&self, attr_name: &str) -> Result<bool> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        let mut value: c_uchar = 0;
        unsafe {
            tf::TF_OperationGetAttrBool(
                self.inner,
                c_attr_name.as_ptr(),
                &mut value,
                status.inner(),
            );
        }
        if !status.is_ok() {
            return Err(status);
        }
        Ok(value != 0)
    }

    /// Get the list of bools in the value of the attribute `attr_name`.
    pub fn get_attr_bool_list(&self, attr_name: &str) -> Result<Vec<bool>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut values: Vec<c_uchar> = Vec::with_capacity(metadata.list_size as usize);
            values.set_len(metadata.list_size as usize);
            tf::TF_OperationGetAttrBoolList(
                self.inner,
                c_attr_name.as_ptr(),
                values.as_mut_ptr(),
                metadata.list_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            #[allow(trivial_numeric_casts)]
            Ok(values.iter().map(|f| *f != 0).collect())
        }
    }

    /// Returns the value of the attribute `attr_name`.
    pub fn get_attr_type(&self, attr_name: &str) -> Result<DataType> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        let mut value: tf::TF_DataType = tf::TF_FLOAT;
        unsafe {
            tf::TF_OperationGetAttrType(
                self.inner,
                c_attr_name.as_ptr(),
                &mut value,
                status.inner(),
            );
        }
        if !status.is_ok() {
            return Err(status);
        }
        Ok(DataType::from_c(value))
    }

    /// Get the list of types in the value of the attribute `attr_name`.
    pub fn get_attr_type_list(&self, attr_name: &str) -> Result<Vec<DataType>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut values: Vec<tf::TF_DataType> = Vec::with_capacity(metadata.list_size as usize);
            values.set_len(metadata.list_size as usize);
            tf::TF_OperationGetAttrTypeList(
                self.inner,
                c_attr_name.as_ptr(),
                values.as_mut_ptr(),
                metadata.list_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            Ok(values.iter().map(|x| DataType::from_c(*x)).collect())
        }
    }

    /// Returns the value of the attribute `attr_name`.
    pub fn get_attr_shape(&self, attr_name: &str) -> Result<Shape> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            if metadata.total_size == -1 {
                return Ok(Shape(None));
            }
            let mut v: Vec<i64> = Vec::with_capacity(metadata.total_size as usize);
            v.set_len(metadata.total_size as usize);
            tf::TF_OperationGetAttrShape(
                self.inner,
                c_attr_name.as_ptr(),
                v.as_mut_ptr(),
                metadata.total_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            Ok(Shape(Some(
                v.iter()
                    .map(|x| if *x < 0 { None } else { Some(*x) })
                    .collect(),
            )))
        }
    }

    /// Get the list of shapes in the value of the attribute `attr_name`.
    pub fn get_attr_shape_list(&self, attr_name: &str) -> Result<Vec<Shape>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut storage: Vec<i64> = Vec::with_capacity(metadata.total_size as usize);
            storage.set_len(metadata.total_size as usize);
            let mut dims: Vec<*mut i64> = Vec::with_capacity(metadata.list_size as usize);
            let mut num_dims: Vec<c_int> = Vec::with_capacity(metadata.list_size as usize);
            tf::TF_OperationGetAttrShapeList(
                self.inner,
                c_attr_name.as_ptr(),
                dims.as_mut_ptr(),
                num_dims.as_mut_ptr(),
                metadata.list_size as i32,
                storage.as_mut_ptr(),
                metadata.total_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            dims.set_len(metadata.list_size as usize);
            num_dims.set_len(metadata.list_size as usize);
            let mut shapes = Vec::with_capacity(metadata.list_size as usize);
            for i in 0..metadata.list_size as usize {
                shapes.push(Shape(if num_dims[i] == -1 {
                    None
                } else {
                    let mut v = Vec::new();
                    for j in 0..num_dims[i] {
                        v.push(match *dims[i].offset(j as isize) {
                            -1 => None,
                            x => Some(x),
                        });
                    }
                    Some(v)
                }));
            }
            Ok(shapes)
        }
    }

    /// Returns the binary-serialized TensorShapeProto value of the attribute
    /// `attr_name`.
    pub fn get_attr_tensor_shape_proto(&self, attr_name: &str) -> Result<Vec<u8>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let mut buf = Buffer::<u8>::new_unallocated();
            tf::TF_OperationGetAttrTensorShapeProto(
                self.inner,
                c_attr_name.as_ptr(),
                buf.inner_mut(),
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            Ok(buf.into())
        }
    }

    /// Get the list of binary-serialized TensorShapeProtos in the value of the
    /// attribute `attr_name`.
    pub fn get_attr_tensor_shape_proto_list(&self, attr_name: &str) -> Result<Vec<Vec<u8>>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut c_buffers = Vec::with_capacity(metadata.list_size as usize);
            for _ in 0..metadata.list_size {
                c_buffers.push(ptr::null_mut());
            }
            tf::TF_OperationGetAttrTensorShapeProtoList(
                self.inner,
                c_attr_name.as_ptr(),
                c_buffers.as_mut_ptr(),
                metadata.list_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            Ok(c_buffers
                .iter()
                .map(|b| Buffer::from_c(*b, true).into())
                .collect())
        }
    }

    /// Returns the value of the attribute `attr_name`. Returns an error if the
    /// type of the tensor value does not match the type of the generic
    /// argument.
    pub fn get_attr_tensor<T: TensorType>(&self, attr_name: &str) -> Result<Tensor<T>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let mut c_tensor: *mut tf::TF_Tensor = ptr::null_mut();
            tf::TF_OperationGetAttrTensor(
                self.inner,
                c_attr_name.as_ptr(),
                &mut c_tensor,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            match Tensor::from_tf_tensor(c_tensor) {
                None => Err(invalid_arg!("Tensor types do not match")),
                Some(t) => Ok(t),
            }
        }
    }

    /// Get the list of tensors in the value of the attribute `attr_name`.
    /// Returns an error if the type of the tensor value does not match the type
    /// of the generic argument.
    pub fn get_attr_tensor_list<T: TensorType>(&self, attr_name: &str) -> Result<Vec<Tensor<T>>> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let metadata =
                tf::TF_OperationGetAttrMetadata(self.inner, c_attr_name.as_ptr(), status.inner());
            if !status.is_ok() {
                return Err(status);
            }
            let mut c_tensors = Vec::with_capacity(metadata.list_size as usize);
            for _ in 0..metadata.list_size {
                c_tensors.push(ptr::null_mut());
            }
            tf::TF_OperationGetAttrTensorList(
                self.inner,
                c_attr_name.as_ptr(),
                c_tensors.as_mut_ptr(),
                metadata.list_size as c_int,
                status.inner(),
            );
            if !status.is_ok() {
                return Err(status);
            }
            c_tensors
                .iter()
                .map(|t| match Tensor::from_tf_tensor(*t) {
                    None => Err(invalid_arg!("Tensor types do not match")),
                    Some(t) => Ok(t),
                })
                .collect()
        }
    }

    /// Returns the binary-serialized AttrValue proto representation of the
    /// value of the `attr_name` attr.
    pub fn get_attr_value_proto(&self, attr_name: &str) -> Result<Vec<u8>> {
        let status = Status::new();
        let attr_name_cstr = CString::new(attr_name)?;
        unsafe {
            let mut buf = Buffer::new_unallocated();
            tf::TF_OperationGetAttrValueProto(
                self.inner,
                attr_name_cstr.as_ptr(),
                buf.inner_mut(),
                status.inner,
            );
            status.into_result()?;
            Ok(buf.into())
        }
    }

    pub(crate) fn inner(&self) -> *mut tf::TF_Operation {
        self.inner
    }
}

impl Into<Output> for Operation {
    /// Creates an Output for index 0.
    fn into(self) -> Output {
        Output {
            operation: self,
            index: 0,
        }
    }
}

////////////////////////

/// A `Input` is one end of a graph edge.
/// It holds an operation and an index into the inputs of that operation.
#[derive(Debug, Copy, Clone)]
pub struct Input<'a> {
    /// Operation the edge connects to.
    pub operation: &'a Operation,

    /// Index into either the inputs of the operation.
    pub index: c_int,
}

////////////////////////

/// A `Output` is one end of a graph edge.
/// It holds an operation and an index into the outputs of that operation.
#[derive(Debug, Clone)]
pub struct Output {
    /// Operation the edge connects to.
    pub operation: Operation,

    /// Index into either the outputs of the operation.
    pub index: c_int,
}

impl Output {
    pub(crate) fn to_c(&self) -> tf::TF_Output {
        tf::TF_Output {
            oper: self.operation.inner,
            index: self.index,
        }
    }

    pub(crate) fn from_c(graph: &Graph, output: &tf::TF_Output) -> Self {
        Output {
            operation: Operation {
                inner: output.oper,
                gimpl: graph.gimpl.clone(),
            },
            index: output.index,
        }
    }

    pub(crate) fn from_c_optional(graph: &Graph, output: &tf::TF_Output) -> Option<Self> {
        if output.oper.is_null() {
            None
        } else {
            Some(Output {
                operation: Operation {
                    inner: output.oper,
                    gimpl: graph.gimpl.clone(),
                },
                index: output.index,
            })
        }
    }
}

////////////////////////

/// An `OperationDescription` is an `Operation` in the process of being built
/// (i.e. the builder pattern).
///
/// An `OperationDescription` is required to be finished before the graph
/// goes out of scope,
/// so `finish()` will be called on drop if it was not already called.
#[derive(Debug)]
pub struct OperationDescription<'a> {
    inner: *mut tf::TF_OperationDescription,
    // This keeps self from outliving the Graph, which is required by
    // the docs on TF_NewOperation.
    graph: &'a Graph,
    finished: bool,
}

impl<'a> Drop for OperationDescription<'a> {
    fn drop(&mut self) {
        if !self.finished {
            unsafe {
                // TF_NewOperation requires us to make sure TF_FinishOperation is called before the
                // graph is deleted.  Combined with guaranteeing that OperationDescription does
                // not outlive Graph, this ensures that the contract is held.
                let status = tf::TF_NewStatus();
                tf::TF_FinishOperation(self.inner, status);
                tf::TF_DeleteStatus(status);
            }
        }
    }
}

impl<'a> OperationDescription<'a> {
    /// Builds the operation and adds it to the graph.
    pub fn finish(mut self) -> Result<Operation> {
        self.finished = true; // used by the drop code
        let mut status = Status::new();
        let operation = unsafe { tf::TF_FinishOperation(self.inner, status.inner()) };
        if status.is_ok() {
            Ok(Operation {
                inner: operation,
                gimpl: self.graph.gimpl.clone(),
            })
        } else {
            Err(status)
        }
    }

    /// Sets the preferred device.
    /// The empty string means unconstrained.
    pub fn set_device(&mut self, device: &str) -> std::result::Result<(), NulError> {
        let c_device = CString::new(device)?;
        unsafe {
            tf::TF_SetDevice(self.inner, c_device.as_ptr());
        }
        Ok(())
    }

    /// Adds an input to this operation.
    ///
    /// The index in the port is an index into the source operation's output array.
    pub fn add_input<I: Into<Output>>(&mut self, input: I) {
        unsafe {
            tf::TF_AddInput(self.inner, input.into().to_c());
        }
    }

    /// Adds multiple inputs to this operation.
    ///
    /// The index in the ports is an index into the source operation's output array.
    pub fn add_input_list(&mut self, inputs: &[Output]) {
        let c_inputs: Vec<tf::TF_Output> = inputs.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TF_AddInputList(self.inner, c_inputs.as_ptr(), c_inputs.len() as c_int);
        }
    }

    /// Adds a control input.
    pub fn add_control_input(&mut self, input: &Operation) {
        unsafe {
            tf::TF_AddControlInput(self.inner, input.inner);
        }
    }

    /// Sets the value of a string attribute.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_string(
        &mut self,
        attr_name: &str,
        value: &str,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value = value.as_bytes();
        unsafe {
            tf::TF_SetAttrString(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr() as *const std_c_void,
                c_value.len() as size_t,
            );
        }
        Ok(())
    }

    /// Sets the value of an attribute which holds a list of strings.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_string_list<S: AsRef<str>>(
        &mut self,
        attr_name: &str,
        value: &[S],
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let bytes: Vec<&[u8]> = value.iter().map(|x| x.as_ref().as_bytes()).collect();
        let ptrs: Vec<*const c_void> = bytes.iter().map(|x| x.as_ptr() as *const c_void).collect();
        let lens: Vec<size_t> = bytes.iter().map(|x| x.len() as size_t).collect();
        unsafe {
            tf::TF_SetAttrStringList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_ptr() as *const *const std_c_void,
                lens.as_ptr(),
                ptrs.len() as c_int,
            );
        }
        Ok(())
    }

    /// Sets the value of a function attribute.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_func_name(
        &mut self,
        attr_name: &str,
        value: &str,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value = value.as_bytes();
        unsafe {
            tf::TF_SetAttrFuncName(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr() as *const c_char,
                c_value.len() as size_t,
            );
        }
        Ok(())
    }

    /// Sets an int-valued attribute.
    pub fn set_attr_int(
        &mut self,
        attr_name: &str,
        value: i64,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrInt(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of ints.
    pub fn set_attr_int_list(
        &mut self,
        attr_name: &str,
        value: &[i64],
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrIntList(
                self.inner,
                c_attr_name.as_ptr(),
                value.as_ptr(),
                value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a float-valued attribute.
    pub fn set_attr_float(
        &mut self,
        attr_name: &str,
        value: f32,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of floats.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_float_list(
        &mut self,
        attr_name: &str,
        value: &[f32],
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        // Allow trivial_numeric_casts here because f32 is not necessarily equal to c_float.
        let c_value: Vec<c_float> = value.iter().map(|x| *x as c_float).collect();
        unsafe {
            tf::TF_SetAttrFloatList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a boolean-valued attribute.
    pub fn set_attr_bool(
        &mut self,
        attr_name: &str,
        value: bool,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrBool(self.inner, c_attr_name.as_ptr(), if value { 1 } else { 0 });
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of booleans.
    pub fn set_attr_bool_list(
        &mut self,
        attr_name: &str,
        value: &[bool],
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<c_uchar> = value.iter().map(|x| if *x { 1 } else { 0 }).collect();
        unsafe {
            tf::TF_SetAttrBoolList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as c_int,
            );
        }
        Ok(())
    }

    /// Sets a type-valued attribute.
    pub fn set_attr_type(
        &mut self,
        attr_name: &str,
        value: DataType,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of types.
    pub fn set_attr_type_list(
        &mut self,
        attr_name: &str,
        value: &[DataType],
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<tf::TF_DataType> = value.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TF_SetAttrTypeList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a shape-valued attribute.
    pub fn set_attr_shape(
        &mut self,
        attr_name: &str,
        value: &Shape,
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            match value.0 {
                None => tf::TF_SetAttrShape(self.inner, c_attr_name.as_ptr(), ptr::null(), -1),
                Some(ref dims) => {
                    let c_dims: Vec<i64> = dims
                        .iter()
                        .map(|x| match *x {
                            Some(d) => d,
                            None => -1,
                        })
                        .collect();
                    tf::TF_SetAttrShape(
                        self.inner,
                        c_attr_name.as_ptr(),
                        c_dims.as_ptr(),
                        c_dims.len() as i32,
                    );
                }
            }
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of shapes.
    pub fn set_attr_shape_list(
        &mut self,
        attr_name: &str,
        value: &[Shape],
    ) -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        // Convert Option<i64> in each shape to i64 with None becoming -1.
        let c_dims: Vec<Option<Vec<i64>>> = value
            .iter()
            .map(|x| match x.0 {
                None => None,
                Some(ref dims) => Some(
                    dims.iter()
                        .map(|x| match *x {
                            None => -1,
                            Some(d) => d,
                        })
                        .collect(),
                ),
            })
            .collect();
        let ptrs: Vec<*const i64> = c_dims
            .iter()
            .map(|x| match *x {
                None => ptr::null(),
                Some(ref dims) => dims.as_ptr(),
            })
            .collect();
        let lens: Vec<c_int> = value
            .iter()
            .map(|x| match x.0 {
                None => -1,
                Some(ref dims) => dims.len() as c_int,
            })
            .collect();
        unsafe {
            tf::TF_SetAttrShapeList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_ptr(),
                lens.as_ptr(),
                ptrs.len() as c_int,
            );
        }
        Ok(())
    }

    /// Sets an attribute with a `TensorShapeProto` protobuf.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_tensor_shape_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrTensorShapeProto(
                self.inner,
                c_attr_name.as_ptr(),
                value.as_ptr() as *const std_c_void,
                value.len() as size_t,
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Sets an attribute with an array of `TensorShapeProto` protobufs.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_tensor_shape_proto_list<T: AsRef<[u8]>>(
        &mut self,
        attr_name: &str,
        value: &[T],
    ) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let ptrs: Vec<*const c_void> = value
            .iter()
            .map(|x| x.as_ref().as_ptr() as *const c_void)
            .collect();
        let lens: Vec<size_t> = value.iter().map(|x| x.as_ref().len() as size_t).collect();
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrTensorShapeProtoList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_ptr() as *const *const std_c_void,
                lens.as_ptr(),
                ptrs.len() as c_int,
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Sets a tensor-valued attribute.
    pub fn set_attr_tensor<T: TensorType>(
        &mut self,
        attr_name: &str,
        value: Tensor<T>,
    ) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrTensor(
                self.inner,
                c_attr_name.as_ptr(),
                value.inner()?,
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Sets an attribute which holds an array of tensors.
    pub fn set_attr_tensor_list<I, T>(&mut self, attr_name: &str, value: I) -> Result<()>
    where
        I: IntoIterator<Item = Tensor<T>>,
        T: TensorType,
    {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            // These have to stay alive durng the TF_SetAttrTensorList call.
            let tensors: Vec<_> = value.into_iter().collect();
            let maybe_ptrs: Result<_> = tensors.iter().map(|x| x.inner()).collect();
            let ptrs: Vec<*mut tf::TF_Tensor> = maybe_ptrs?;
            tf::TF_SetAttrTensorList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_ptr() as *const *const tf::TF_Tensor,
                ptrs.len() as c_int,
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Sets an attribute with an `AttrValue` proto.
    #[deprecated(since = "0.7.0", note = "Use set_attr_value_proto instead.")]
    pub fn set_attr_to_attr_value_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
        self.set_attr_value_proto(attr_name, value)
    }

    /// Sets an attribute with an `AttrValue` proto.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_value_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrValueProto(
                self.inner,
                c_attr_name.as_ptr(),
                value.as_ptr() as *const std_c_void,
                // Allow trivial_numeric_casts because usize is not
                // necessarily size_t.
                value.len() as size_t,
                status.inner(),
            );
        }
        status.into_result()
    }
}

////////////////////////

/// Options that can be passed during function creation.
#[derive(Debug)]
#[allow(missing_copy_implementations)]
pub struct FunctionOptions {
    inner: *mut tf::TF_FunctionOptions,
}

impl FunctionOptions {
    /// Creates a blank set of options.
    pub fn new() -> Self {
        FunctionOptions {
            inner: ptr::null_mut(), // TODO: Use real options when they become available
        }
    }
}

////////////////////////

/// Function is a grouping of operations with defined inputs and outputs.
/// Once created and added to graphs, functions can be invoked by creating an
/// operation whose operation type matches the function name.
#[derive(Debug)]
pub struct Function {
    inner: *mut tf::TF_Function,
}

impl_drop!(Function, TF_DeleteFunction);

impl Function {
    /// Returns a serialized representation of the function (as a FunctionDef
    /// protocol message).
    ///
    /// May fail on very large graphs in the future.
    pub fn to_function_def(&self) -> Result<Vec<u8>> {
        let status = Status::new();
        unsafe {
            let mut buf = Buffer::from_ptr(ptr::null_mut(), 0);
            tf::TF_FunctionToFunctionDef(self.inner, buf.inner_mut(), status.inner);
            status.into_result()?;
            Ok(buf.into())
        }
    }

    /// Construct and return the function whose FunctionDef representation is
    /// serialized in `proto`. Returns a newly created `Function` instance.
    pub fn import_function_def(proto: &[u8]) -> Result<Function> {
        let status = Status::new();
        unsafe {
            let inner = tf::TF_FunctionImportFunctionDef(
                proto.as_ptr() as *const std_c_void,
                proto.len(),
                status.inner,
            );
            status.into_result()?;
            Ok(Function { inner })
        }
    }

    /// Sets function attribute named `attr_name` to value stored in `proto`. If
    /// this attribute is already set to another value, it is overriden. `proto`
    /// should be a sequence of bytes representing a binary serialization of an
    /// AttrValue protocol buffer.
    pub fn set_attr_value_proto(&mut self, attr_name: &str, proto: &[u8]) -> Result<()> {
        let status = Status::new();
        let attr_name_cstr = CString::new(attr_name)?;
        unsafe {
            tf::TF_FunctionSetAttrValueProto(
                self.inner,
                attr_name_cstr.as_ptr(),
                proto.as_ptr() as *const std_c_void,
                proto.len(),
                status.inner,
            );
        }
        status.into_result()
    }

    /// Returns the binary-serialized AttrValue proto representation of the
    /// value of the `attr_name` attr of the function. If `attr_name` attribute
    /// is not present, returns an error.
    pub fn get_attr_value_proto(&self, attr_name: &str) -> Result<Vec<u8>> {
        let status = Status::new();
        let attr_name_cstr = CString::new(attr_name)?;
        unsafe {
            let mut buf = Buffer::from_ptr(ptr::null_mut(), 0);
            tf::TF_FunctionGetAttrValueProto(
                self.inner,
                attr_name_cstr.as_ptr(),
                buf.inner_mut(),
                status.inner,
            );
            status.into_result()?;
            Ok(buf.into())
        }
    }

    /// Returns the name of the graph function.
    pub fn get_name(&self) -> std::result::Result<String, Utf8Error> {
        unsafe {
            CStr::from_ptr(tf::TF_FunctionName(self.inner))
                .to_str()
                .map(|s| s.to_string())
        }
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::super::DataType;
    use super::super::Shape;
    use super::*;

    fn add_operation(g: &mut Graph) {
        g.new_operation("Variable", "foo").unwrap();
    }

    fn add(g: &mut Graph, op1: Operation, op2: Operation, name: &str) -> Result<Operation> {
        let mut nd = g.new_operation("Add", name)?;
        nd.add_input(op1);
        nd.add_input(op2);
        nd.finish()
    }

    fn multiply(g: &mut Graph, op1: Operation, op2: Operation, name: &str) -> Result<Operation> {
        let mut nd = g.new_operation("Mul", name)?;
        nd.add_input(op1);
        nd.add_input(op2);
        nd.finish()
    }

    #[test]
    fn smoke() {
        let mut g = Graph::new();
        add_operation(&mut g);
        let operation = {
            let mut nd = g.new_operation("Variable", "foo").unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
            nd.finish().unwrap()
        };
        let mut nd2 = g.new_operation("Variable", "foo2").unwrap();
        nd2.set_attr_type("dtype", DataType::Float).unwrap();
        nd2.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
        let operation2 = nd2.finish().unwrap();
        assert_eq!("foo", operation.name().unwrap());
        assert_eq!("foo2", operation2.name().unwrap());
    }

    #[test]
    fn test_import_graph_def() {
        let mut g = Graph::new();
        let opts = ImportGraphDefOptions::new();
        // An empty array is a valid proto, since all fields are optional.
        let status = g.import_graph_def(&[], &opts);
        assert!(status.is_ok());
    }

    #[test]
    fn test_get_tensor_shape() {
        fn constant<T: TensorType>(graph: &mut Graph, name: &str, value: Tensor<T>) -> Operation {
            let mut c = graph.new_operation("Const", name).unwrap();
            c.set_attr_tensor("value", value).unwrap();
            c.set_attr_type("dtype", T::data_type()).unwrap();
            c.finish().unwrap()
        }

        let mut graph = Graph::new();
        let x_init = Tensor::<i32>::new(&[3, 3]);
        let x = constant(&mut graph, "x/assign_0", x_init);
        assert_eq!(1, x.num_outputs());
        assert_eq!(x.output_type(0), DataType::Int32);
        let dims = graph.num_dims(x.clone()).unwrap();
        assert_eq!(dims, 2);
        let shape = graph.tensor_shape(x.clone()).unwrap();
        assert_eq!(shape, Shape(Some(vec![Some(3_i64), Some(3_i64)])));
    }

    #[test]
    fn graph_to_function() {
        let mut g = Graph::new();
        let x = {
            let mut nd = g.new_operation("Placeholder", "x").unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
            nd.finish().unwrap()
        };
        let two = {
            let mut nd = g.new_operation("Const", "two").unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            let mut value = Tensor::new(&[1]);
            value[0] = 2.0f32;
            nd.set_attr_tensor("value", value).unwrap();
            nd.finish().unwrap()
        };
        let y = multiply(&mut g, two.clone(), x.clone(), "y").unwrap();
        let opers = vec![&y];
        let inputs = vec![x.clone().into(), two.clone().into()];
        let outputs = vec![y.clone().into()];
        let output_names = vec!["result"];
        let description = "Multiplies by 2";
        let opts = FunctionOptions::new();
        let f = g
            .to_function(
                "times_two",
                false,
                Some(&opers),
                &inputs,
                &outputs,
                Some(&output_names),
                &opts,
                Some(description),
            )
            .unwrap();
        assert_eq!("times_two", f.get_name().unwrap());
        let mut g2 = Graph::new();
        assert_eq!(0, g2.num_functions());
        assert_eq!(0, g2.get_functions().unwrap().len());
        g2.copy_function(&f, None).unwrap();
        assert_eq!(1, g2.num_functions());
        assert_eq!(1, g2.get_functions().unwrap().len());
    }

    // This test checks that Operation::get_attr_* returns the value passed in
    // by OperationDescription::set_attr_*.  It's long and tedious because we
    // need to create several different ops to cover all the different types,
    // and the ops have requirements that have to be set up, first.  Once we can
    // define our own ops, we may be able to just define a single op with
    // attributes for all of the types.
    #[test]
    #[allow(trivial_casts)] // so we can do assert_eq!(slice, &some_vec as &[_])
    fn operation_attributes() {
        let mut g = Graph::new();

        let shape = Shape(Some(vec![None, Some(3)]));
        let variable_op = {
            let mut nd = g.new_operation("Variable", "Variable").unwrap();
            nd.set_attr_type("dtype", DataType::Int32).unwrap();
            nd.set_attr_shape("shape", &shape).unwrap();
            nd.set_attr_string("shared_name", "bar").unwrap();
            nd.finish().unwrap()
        };
        assert_eq!("bar", variable_op.get_attr_string("shared_name").unwrap());
        assert_eq!(DataType::Int32, variable_op.get_attr_type("dtype").unwrap());
        assert_eq!(shape, variable_op.get_attr_shape("shape").unwrap());

        let op = {
            let mut nd = g
                .new_operation("Variable", "Variable_unknown_rank")
                .unwrap();
            nd.set_attr_type("dtype", DataType::Int32).unwrap();
            nd.set_attr_shape("shape", &Shape(None)).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(Shape(None), op.get_attr_shape("shape").unwrap());

        let value = Tensor::<i32>::new(&[1, 3]).with_values(&[1, 2, 3]).unwrap();
        let const_op = {
            let mut nd = g.new_operation("Const", "Const").unwrap();
            nd.set_attr_tensor("value", value.clone()).unwrap();
            nd.set_attr_type("dtype", DataType::Int32).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(value, const_op.get_attr_tensor("value").unwrap());

        let op = {
            let mut nd = g.new_operation("Assign", "Assign").unwrap();
            nd.add_input(variable_op.clone());
            nd.add_input(variable_op.clone());
            nd.set_attr_bool("validate_shape", true).unwrap();
            nd.set_attr_bool("use_locking", false).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(true, op.get_attr_bool("validate_shape").unwrap());
        assert_eq!(false, op.get_attr_bool("use_locking").unwrap());

        let op = {
            let variable_op = {
                let mut nd = g.new_operation("Variable", "MaxPool_in1").unwrap();
                nd.set_attr_type("dtype", DataType::Int32).unwrap();
                nd.set_attr_shape(
                    "shape",
                    &Shape(Some(vec![Some(5), Some(5), Some(5), Some(5)])),
                )
                .unwrap();
                nd.finish().unwrap()
            };
            let mut nd = g.new_operation("MaxPool", "MaxPool").unwrap();
            nd.add_input(variable_op);
            nd.set_attr_int_list("ksize", &[1, 2, 3, 4]).unwrap();
            nd.set_attr_int_list("strides", &[1, 1, 1, 1]).unwrap();
            nd.set_attr_string("padding", "VALID").unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(
            &[1, 2, 3, 4],
            &op.get_attr_int_list("ksize").unwrap() as &[i64]
        );

        let op = {
            let mut nd = g.new_operation("TensorSummary", "TensorSummary").unwrap();
            nd.add_input(variable_op.clone());
            nd.set_attr_string_list("labels", &["foo", "bar"]).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(
            &["foo".to_string(), "bar".to_string()],
            &op.get_attr_string_list("labels").unwrap() as &[_]
        );

        let op = {
            let mut nd = g
                .new_operation("ApproximateEqual", "ApproximateEqual")
                .unwrap();
            nd.add_input(variable_op.clone());
            nd.add_input(variable_op.clone());
            nd.set_attr_float("tolerance", 3.14).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(3.14, op.get_attr_float("tolerance").unwrap());

        let op = {
            let mut nd = g.new_operation("Bucketize", "Bucketize").unwrap();
            nd.add_input(variable_op.clone());
            nd.set_attr_float_list("boundaries", &[0.1, 2.3]).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(
            &[0.1f32, 2.3],
            &op.get_attr_float_list("boundaries").unwrap() as &[_]
        );

        let shape_list = &[
            Shape(None),
            Shape(Some(vec![])),
            Shape(Some(vec![None])),
            Shape(Some(vec![Some(1)])),
        ];
        let op = {
            let mut nd = g
                .new_operation("RandomShuffleQueue", "RandomShuffleQueue")
                .unwrap();
            nd.set_attr_shape_list("shapes", shape_list).unwrap();
            nd.set_attr_type_list("component_types", &[DataType::Float, DataType::Int32])
                .unwrap();
            nd.set_attr_int("seed", 42).unwrap();
            nd.finish().unwrap()
        };
        assert_eq!(
            shape_list,
            &op.get_attr_shape_list("shapes").unwrap() as &[_]
        );
        assert_eq!(
            &[DataType::Float, DataType::Int32],
            &op.get_attr_type_list("component_types").unwrap() as &[_]
        );
        assert_eq!(42, op.get_attr_int("seed").unwrap());

        // TODO: Support get_attr_*/set_attr_*:
        // - bool_list
        // - tensor_list
        // - tensor_shape_proto
        // - tensor_shape_proto_list
        // - value_proto
        // - func_name
        // The protos are tricky because we don't currently support proto
        // serialization/deserialization, and bool_list and tensor_list (a.k.a.
        // list(bool) and list(tensor)) don't seem to be used for any standard
        // ops. TF_GetAttrFuncName doesn't exist yet.
    }

    // Returns a serialized GraphDef proto with variables "a" and "b" and op "a_times_b".
    fn graph_def() -> Vec<u8> {
        let mut g = Graph::new();
        let a = {
            let mut nd = g.new_operation("Variable", "a").unwrap();
            nd.set_attr_type("dtype", DataType::Int32).unwrap();
            nd.set_attr_shape("shape", &Shape(None)).unwrap();
            nd.finish().unwrap()
        };
        let b = {
            let mut nd = g.new_operation("Variable", "b").unwrap();
            nd.set_attr_type("dtype", DataType::Int32).unwrap();
            nd.set_attr_shape("shape", &Shape(None)).unwrap();
            nd.finish().unwrap()
        };
        multiply(&mut g, a, b, "a_times_b").unwrap();
        g.graph_def().unwrap()
    }

    #[test]
    fn import_graph_def_uniquify_names() {
        let mut g = Graph::new();
        let mut opts = ImportGraphDefOptions::new();
        g.import_graph_def(&graph_def(), &opts).unwrap();
        opts.set_uniquify_names(true);
        g.import_graph_def(&graph_def(), &opts).unwrap();
        g.operation_by_name_required("a_1").unwrap();
    }

    #[test]
    fn import_graph_def_uniquify_prefix() {
        let mut g = Graph::new();
        let mut opts = ImportGraphDefOptions::new();
        opts.set_prefix("prefix").unwrap();
        g.import_graph_def(&graph_def(), &opts).unwrap();
        opts.set_uniquify_prefix(true);
        g.import_graph_def(&graph_def(), &opts).unwrap();
        g.operation_by_name_required("prefix_1/a").unwrap();
    }

    #[test]
    fn import_graph_def_set_default_device() {
        let mut g = Graph::new();
        let mut opts = ImportGraphDefOptions::new();
        opts.set_default_device("fake_device").unwrap();
        g.import_graph_def(&graph_def(), &opts).unwrap();
        assert_eq!(
            g.operation_by_name_required("a").unwrap().device().unwrap(),
            "fake_device"
        );
    }

    #[test]
    fn import_graph_def_results_return_outputs() {
        let mut g = Graph::new();
        let mut opts = ImportGraphDefOptions::new();
        assert_eq!(opts.num_return_outputs(), 0);
        opts.add_return_output("a_times_b", 0).unwrap();
        assert_eq!(opts.num_return_outputs(), 1);
        let result = g
            .import_graph_def_with_results(&graph_def(), &opts)
            .unwrap();
        let ops = result.return_outputs();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].operation.name().unwrap(), "a_times_b");
        assert_eq!(ops[0].index, 0);
    }

    #[test]
    fn import_graph_def_results_return_operations() {
        let mut g = Graph::new();
        let mut opts = ImportGraphDefOptions::new();
        assert_eq!(opts.num_return_operations(), 0);
        opts.add_return_operation("a_times_b").unwrap();
        assert_eq!(opts.num_return_operations(), 1);
        let result = g
            .import_graph_def_with_results(&graph_def(), &opts)
            .unwrap();
        let ops = result.return_operations();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].name().unwrap(), "a_times_b");
    }

    #[test]
    fn import_graph_def_results_missing_unused_input_mappings() {
        let mut g = Graph::new();
        let op = {
            let mut nd = g.new_operation("Variable", "foo").unwrap();
            nd.set_attr_type("dtype", DataType::Int32).unwrap();
            nd.set_attr_shape("shape", &Shape(None)).unwrap();
            nd.finish().unwrap()
        };
        let output = op.into();
        let mut opts = ImportGraphDefOptions::new();
        opts.add_input_mapping("bar", 3, &output).unwrap();
        // An empty array is a valid proto, since all fields are optional.
        let result = g.import_graph_def_with_results(&[], &opts).unwrap();
        let missing = result.missing_unused_input_mappings().unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].0, "bar");
        assert_eq!(missing[0].1, 3);
    }

    #[test]
    fn import_graph_def_with_return_outputs() {
        let mut g = Graph::new();
        let mut opts = ImportGraphDefOptions::new();
        assert_eq!(opts.num_return_outputs(), 0);
        opts.add_return_output("a_times_b", 0).unwrap();
        assert_eq!(opts.num_return_outputs(), 1);
        let ops = g
            .import_graph_def_with_return_outputs(&graph_def(), &opts)
            .unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].operation.name().unwrap(), "a_times_b");
        assert_eq!(ops[0].index, 0);
    }

    #[test]
    fn graph_get_op_def() {
        let g = Graph::new();
        // We don't want to compare the actual proto because it may change across releases.
        assert!(g.get_op_def("Const").unwrap().len() > 0);
    }

    #[test]
    fn graph_versions() {
        let g = Graph::new();
        // We don't want to compare the actual proto because it may change across releases.
        assert!(g.versions().unwrap().len() > 0);
    }

    #[test]
    fn graph_generate_operation_name() {
        let mut g = Graph::new();
        for i in 0..5 {
            assert_eq!(i, g.generate_operation_name("foo_{}").unwrap());
            let mut nd = g
                .new_operation("Placeholder", &format!("foo_{}", i))
                .unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
            nd.finish().unwrap();
        }
    }

    #[test]
    fn graph_add_gradients() {
        // TODO: Add an integration test to verify that the gradient behaves as expected.
        for (prefix, expected_prefix) in &[
            (Some("arbitrary_prefix"), "arbitrary_prefix/"),
            (None, "gradients/"),
        ] {
            let mut g = Graph::new();
            let x = {
                let mut nd = g.new_operation("Placeholder", "x").unwrap();
                nd.set_attr_type("dtype", DataType::Float).unwrap();
                nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
                nd.finish().unwrap()
            };
            let y = {
                let mut nd = g.new_operation("Placeholder", "y").unwrap();
                nd.set_attr_type("dtype", DataType::Float).unwrap();
                nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
                nd.finish().unwrap()
            };
            let x_squared = multiply(&mut g, x.clone(), x.clone(), "x_squared").unwrap();
            let x_times_y = multiply(&mut g, x.clone(), y.clone(), "x_times_y").unwrap();
            let x_plus_y = add(&mut g, x.clone(), y.clone(), "x_plus_y").unwrap();
            // y_outs and x_outs are intentionally different lengths, so we can test that the lengths line up properly.
            let y_outs = vec![x_squared.into(), x_times_y.into(), x_plus_y.into()];
            let x_outs = vec![x.into(), y.into()];
            let dy = g.add_gradients(*prefix, &y_outs, &x_outs, None).unwrap();
            assert_eq!(dy.len(), 2);
            for d in dy {
                let d = d.unwrap();
                assert_eq!(d.index, 0);
                let name = d.operation.name().unwrap();
                assert!(
                    name.starts_with(expected_prefix),
                    "name = {}, expected prefix = {}",
                    name,
                    expected_prefix
                );
            }
        }
    }

    #[test]
    fn graph_add_gradients_stopped_gradient() {
        // TODO: Add an integration test to verify that the gradient behaves as expected.
        for prefix in &[Some("arbitrary_prefix"), None] {
            let mut g = Graph::new();
            let zero = {
                let mut nd = g.new_operation("Const", "zero").unwrap();
                nd.set_attr_type("dtype", DataType::Int32).unwrap();
                nd.set_attr_tensor("value", Tensor::<i32>::from(0)).unwrap();
                nd.finish().unwrap()
            };
            let x = {
                let mut nd = g.new_operation("Placeholder", "x").unwrap();
                nd.set_attr_type("dtype", DataType::Float).unwrap();
                nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
                nd.finish().unwrap()
            };
            let argmax_x = {
                let mut nd = g.new_operation("ArgMax", "argmax_x").unwrap();
                nd.add_input(x.clone());
                nd.add_input(zero);
                nd.finish().unwrap()
            };
            let stopped_gradient = {
                let mut nd = g.new_operation("StopGradient", "stopped").unwrap();
                nd.add_input(argmax_x.clone());
                nd.finish().unwrap()
            };
            let y_outs = vec![stopped_gradient.into()];
            let x_outs = vec![x.into()];
            let dy = g.add_gradients(*prefix, &y_outs, &x_outs, None).unwrap();
            assert_eq!(dy.len(), 1);
            for d in &dy {
                assert!(d.is_none());
            }
        }
    }

    #[test]
    fn graph_add_gradients_no_gradient() {
        // TODO: Add an integration test to verify that the gradient behaves as expected.
        for prefix in &[Some("arbitrary_prefix"), None] {
            let mut g = Graph::new();
            let zero = {
                let mut nd = g.new_operation("Const", "zero").unwrap();
                nd.set_attr_type("dtype", DataType::Int32).unwrap();
                nd.set_attr_tensor("value", Tensor::<i32>::from(0)).unwrap();
                nd.finish().unwrap()
            };
            let x = {
                let mut nd = g.new_operation("Placeholder", "x").unwrap();
                nd.set_attr_type("dtype", DataType::Float).unwrap();
                nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
                nd.finish().unwrap()
            };
            let argmax_x = {
                let mut nd = g.new_operation("ArgMax", "argmax_x").unwrap();
                nd.add_input(x.clone());
                nd.add_input(zero);
                nd.finish().unwrap()
            };
            let y_outs = vec![argmax_x.into()];
            let x_outs = vec![x.into()];
            assert!(g.add_gradients(*prefix, &y_outs, &x_outs, None).is_err());
        }
    }
}
