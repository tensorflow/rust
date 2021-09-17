import tensorflow as tf
from tensorflow.core.framework import op_def_pb2
from google.protobuf import text_format
from termcolor import colored
import re
import textwrap

ops = op_def_pb2.OpList()
text_format.Merge(open("ops.pbtxt").read(), ops)


class Attribute:
    def __init__(self, attr, number_attr_list):

        self.attr = attr
        self.name = self.attr.name

        if self.attr.type == "func":
            raise Exception("Passing functions as arguments is not yet supported")

        # List attributes are defined as 'list(attr)''
        self.type, self.islist = (
            (self.attr.type, False)
            if self.attr.type[:4] != "list"
            else (self.attr.type[5:-1], True)
        )

        self.number_attr = [i for n, i in number_attr_list if self.name == n]
        self.number_attr, self.type = (
            (self.number_attr[0].name, "n_attr")
            if len(self.number_attr)
            else (None, self.type)
        )

        self.default = (
            bool(len(self.attr.default_value.ListFields()))
            and not self.islist
            and self.type not in ["shape", "tensor"]
        )

    def declaration(self):

        # Basic T types attributes are not used
        if self.name == "T":
            return ""

        # Number attributes are infered from others (no need for an argument)
        if self.number_attr is not None:
            return ""

        # Convert from TF types to C++ types
        cpptype = {
            "shape": "&Vec<i64>",
            "int": "i64",
            "float": "f32",
            "string": "&String",
            "type": "DataType",  # Refers to cppflow::datatype
            "bool": "bool",
            "tensor": "&AnyTensor",
        }[self.type]

        # Warp list attributes in a C++ vector
        if self.islist:
            cpptype = cpptype.replace("&", "")  # Not inner reference types
            cpptype = "&Vec<{}>".format(cpptype.replace("const", ""))

        # Get the default value for the attribute
        # Not yet supported for lists
        # Not supported for tensors or shape
        if self.default and not self.islist and self.type not in ["shape", "tensor"]:
            cppdefault = (
                "="
                + {
                    "int": str(self.attr.default_value.i),
                    "bool": str(self.attr.default_value.b).lower(),
                    "string": '"' + str(self.attr.default_value.s)[2:-1] + '"',
                    "float": "{:.4e}".format(self.attr.default_value.f).replace(
                        "inf", "std::numeric_limits<float>::infinity()"
                    ),
                    "type": "static_cast<datatype>({})".format(
                        self.attr.default_value.type
                    ),
                }[self.type]
            )
        else:
            cppdefault = ""

        # datatype name=defaultval
        return self.name.replace("template", "template_arg") + ": " + cpptype

    def code(self):

        # Basic T types attributes are not used
        if self.name == "T":
            return ""

        if self.islist:
            return textwrap.dedent(
                {
                    "string": """
                            let {0}_cstr: Vec<CString> = {0}.into_iter().map(|v| CString::new(v.as_bytes()).unwrap()).collect();
                            let {0}_ptr: Vec<*const c_void> = {0}_cstr.as_slice().into_iter().map(|v| v.as_ptr() as *const c_void).collect();
                            let {0}_sizes: Vec<usize> = {0}_cstr.as_slice().into_iter().map(|v| v.as_bytes().len()).collect();
                            let attr_name = CString::new("{orig:}").unwrap();
                            tf::TFE_OpSetAttrStringList(op, attr_name.as_ptr(), {0}_ptr.as_ptr(), {0}_sizes.as_ptr(), {0}.len() as i32);
                            """,
                    "int": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        tf::TFE_OpSetAttrIntList(op, attr_name.as_ptr(), {0}.as_ptr(), {0}.len() as i32);""",
                    "float": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        tf::TFE_OpSetAttrFloatList(op, attr_name.as_ptr(), {0}.as_ptr(), {0}.len() as i32);""",
                    "bool": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        tf::TFE_OpSetAttrBoolList(op, attr_name.as_ptr(), {0}.into_iter().map(|x| x as u8).collect().as_mut_ptr(), {0}.len() as i32);""",
                    "type": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        let {0}: Vec<tf::TF_DataType> = {0}.into_iter().map(|v| v.to_c()).collect();
                        tf::TFE_OpSetAttrTypeList(op, attr_name.as_ptr(), {0}.as_ptr(), {0}.len() as i32);""",
                    "shape": """
                            let mut {0}_values : Vec<*const i64> = {0}.into_iter().map(|v| v.as_ptr()).collect();
                            let {0}_ndims : Vec<i32> = {0}.into_iter().map(|v| v.len() as i32).collect();
                            let attr_name = CString::new("{orig:}").unwrap();
                            tf::TFE_OpSetAttrShapeList(op, attr_name.as_ptr(), {0}_values.as_mut_ptr(), {0}_ndims.as_ptr(), {0}.len() as i32, status.inner);
                            // status_check(context::get_status());
                            """,
                }[self.type].format(
                    self.name.replace("template", "template_arg"), orig=self.name
                )
            ).replace("\n", "\n    ")

        else:
            return textwrap.dedent(
                {
                    "shape": """
                        let attr_name = CString::new("{orig:}").unwrap();
                          tf::TFE_OpSetAttrShape(op,  attr_name.as_ptr(), {0}.as_ptr(), {0}.len() as i32, status.inner);
                          // status_check(context::get_status());
                           """,
                    "int": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        tf::TFE_OpSetAttrInt(op, attr_name.as_ptr(), {0});""",
                    "float": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        tf::TFE_OpSetAttrFloat(op, attr_name.as_ptr(), {0});""",
                    "string": """
                        let attr_name = CString::new("{orig:}").unwrap();
                        let {0}_cstr = {0}.as_bytes();
                        tf::TFE_OpSetAttrString(op, attr_name.as_ptr(), {0}_cstr.as_ptr() as *mut c_void, {0}.as_bytes().len()+1);""",
                    "type": """
                            let attr_name = CString::new("{orig:}").unwrap();
                            tf::TFE_OpSetAttrType(op, attr_name.as_ptr(), {0}.to_c());""",
                    "bool": """
                            let attr_name = CString::new("{orig:}").unwrap();
                            tf::TFE_OpSetAttrBool(op, attr_name.as_ptr(), {0} as u8);""",
                    "tensor": """
                            let attr_name = CString::new("{orig:}").unwrap();
                           tf::TFE_OpSetAttrTensor(op, attr_name.as_ptr(), {0}.inner().unwrap(), status.inner);
                           // status_check(context::get_status());
                           """,
                    "n_attr": """
                            let attr_name = CString::new("{orig:}").unwrap();
                            tf::TFE_OpSetAttrInt(op, attr_name.as_ptr(), {n_attr:}.len() as i64);""",
                }[self.type].format(
                    self.name.replace("template", "template_arg"),
                    orig=self.name,
                    n_attr=self.number_attr,
                )
            ).replace("\n", "\n    ")


class Operation:
    def __init__(self, op):
        self.op = op

        # More than one output?
        if len(self.op.output_arg) != 1:
            raise Exception("More than one or no output not yet supported")

        self.inputs = [inp for inp in op.input_arg]

        # Number attributes define the length of an input list
        number_attr = [
            (i.number_attr, i) for i in self.inputs if len(i.number_attr) > 0
        ]

        # Attributes
        self.attr_list = sorted(
            [Attribute(a, number_attr) for a in self.op.attr], key=lambda a: a.default
        )

    def code(self):

        # C++ function body
        template = textwrap.dedent(
            """
        pub fn {}<T>(ctx: &Context, {}{}) -> Result<{}>
        where
            T: ToHandle,
            {{
            let status = Status::new();

            unsafe {{
            // Define Op
            let op_name = CString::new("{}").unwrap();
            let op = tf::TFE_NewOp(ctx.inner, op_name.as_ptr(), status.inner);
            
            // Required input arguments
            {}

            // Attributes
            {}

            // Execute Op
            let mut num_output = 1;
            let mut res = [std::ptr::null_mut::<tf::TFE_TensorHandle>()];
            tf::TFE_Execute(
                op,
                res.as_mut_ptr(),
                (&mut num_output) as *mut i32,
                status.inner,
            );
            Ok(TensorHandle {{ inner: res[0] }})
            }}
        }}
        """
        )

        # Add single input template
        add_inputs = textwrap.dedent(
            """
            tf::TFE_OpAddInput(op, {}.to_handle()?.inner, status.inner);
            // status_check(context::get_status());
        """
        ).replace("\n", "\n    ")

        add_inputs_list = textwrap.dedent(
            """
            let mut {0}_handles : Vec<*mut tf::TFE_TensorHandle> = {0}.into_iter().map(|t| t.to_handle().unwrap().inner).collect();
            tf::TFE_OpAddInputList(op, {0}_handles.as_mut_ptr(), {0}.len() as i32, status.inner);
            // status_check(context::get_status());
        """
        ).replace("\n", "\n    ")

        # Return type of the function
        out = "TensorHandle" if len(self.op.output_arg) else "void"

        # snake_case name of the operation
        snk = (
            re.sub(r"(?<!^)(?=[A-Z])", "_", self.op.name)
            .lower()
            .replace("const", "const_tensor")
        )
        snk = snk.replace("mod", "mod_")
        snk = snk.replace("where", "where_")

        # Required input arguments
        inp = ", ".join(
            [
                "{}: &Vec<T>".format(n.name)
                if len(n.number_attr) or len(n.type_list_attr)
                else "{}: T".format(n.name.replace("ref", "ref_"))
                for i, n in enumerate(self.inputs)
            ]
        )

        # Declaration of inpattributes
        atr = ", ".join(a.declaration() for a in self.attr_list if len(a.declaration()))
        atr = (", " + atr) if inp != "" and atr != "" else atr
        atr = atr.replace("type", "type_")

        # Operation original name
        opn = self.op.name

        # Code for input arguments
        inp_code = "\n    ".join(
            add_inputs_list.format(n.name)
            if len(n.number_attr) or len(n.type_list_attr)
            else add_inputs.format(n.name.replace("ref", "ref_"))
            for n in self.inputs
        )

        # Code for attributes
        atr_code = "\n    ".join(a.code() for a in self.attr_list if len(a.code()))
        atr_code = atr_code.replace("type", "type_")

        return template.format(snk, inp, atr, out, opn, inp_code, atr_code)


ops_file = textwrap.dedent(
    """
#![allow(non_snake_case)]
#![allow(missing_docs)]

use crate::eager::Context;
use crate::eager::TensorHandle;
use crate::eager::ToHandle;
use crate::AnyTensor;
use crate::DataType;
use crate::Result;
use crate::Status;
use crate::Tensor;
use crate::TensorType;
use std::ffi::CString;
use std::os::raw::c_void;

use tensorflow_sys as tf;

{}
"""
)


ops_code = ""

num_ops = 0

# All TF C API operations correspond with tf.raw_ops
for op_name in sorted(dir(tf.raw_ops)):
    if not op_name.startswith("_"):

        num_ops += 1

        try:
            # Grab operation definition
            op = [op for op in ops.op if op.name == op_name]
            if len(op) == 0:
                raise Exception("Operation not found")
            #             print(op[0])
            op = Operation(op[0])

            ops_code += op.code()

            # Everything was ok!
            print("{:<50}  [{}]".format(op_name, colored("  Ok  ", "green")))
        except Exception as err:
            print("{:<50}  [{}]".format(op_name, colored("Failed", "red")))
            print("    ", err)


with open("raw_ops.rs", "w") as f:
    f.write(ops_file.format(ops_code))
