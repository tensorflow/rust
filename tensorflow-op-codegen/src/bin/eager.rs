use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::ErrorKind;
use std::io::Write;
use std::path::Path;
use std::result::Result;
use tensorflow_op_codegen::parser;
use tensorflow_op_codegen::protos::OpDef;

use ::protobuf::ProtobufEnum;
use tensorflow_op_codegen::protos::AttrValue_oneof_value;
use tensorflow_op_codegen::protos::OpDef_ArgDef;

#[derive(Clone)]
struct Attr {
    rust_name: String,
    attr_type: String,
    pub c_name: String,
    default_value: Option<AttrValue_oneof_value>,
}

#[derive(Clone)]
struct InputArg {
    pub name: String,
    number_attr: String,
    _type_attr: String,
}
impl InputArg {
    fn has_number_attr(&self) -> bool {
        !self.number_attr.is_empty()
    }
}

fn write_set_attr<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    let c_name = &attr.c_name;
    let rust_name = &attr.rust_name;
    let setter = match attr.attr_type.as_str() {
        "::std::string::String" => format!("op.set_attr_string(\"{}\", value)?", c_name),
        "crate::DataType" => format!("op.set_attr_type(\"{}\", *value)?", c_name),
        "bool" => format!("op.set_attr_bool(\"{}\", *value)?", c_name),
        "f32" => format!("op.set_attr_float(\"{}\", *value)?", c_name),
        "i64" => format!("op.set_attr_int(\"{}\", *value)?", c_name),
        "crate::Shape" => format!("op.set_attr_shape(\"{}\", value)?", c_name),
        "crate::Tensor" => format!("op.set_attr_any_tensor(\"{}\", value)?", c_name),
        "::std::vec::Vec<::std::string::String>" => {
            format!("op.set_attr_string_list(\"{}\", value)?", c_name)
        }
        "::std::vec::Vec<f32>" => {
            format!("op.set_attr_float_list(\"{}\", value)?", c_name)
        }
        "::std::vec::Vec<i64>" => format!("op.set_attr_int_list(\"{}\", value)?", c_name),
        "::std::vec::Vec<crate::DataType>" => {
            format!("op.set_attr_type_list(\"{}\", value)?", c_name)
        }
        "::std::vec::Vec<crate::Shape>" => {
            format!("op.set_attr_shape_list(\"{}\", value)?", c_name)
        }
        ty => panic!("Unrecognized attribute type for {}: {}", attr.rust_name, ty),
    };
    writeln!(
        w,
        "    if let ::std::option::Option::Some(value) = &self.{} {{",
        rust_name
    )?;
    writeln!(w, "        {};", setter)?;
    writeln!(w, "    }}")?;
    Ok(())
}

fn write_short_fn<W: Write>(
    w: &mut W,
    name: &str,
    fn_name: &str,
    input_args: &[&OpDef_ArgDef],
    output_args: &[String],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<InputArg> = input_args
        .iter()
        .map(|arg| InputArg {
            name: escaper.escape(&arg.name),
            number_attr: arg.number_attr.clone(),
            _type_attr: arg.type_attr.clone(),
        })
        .collect();
    write!(w, "/// Shorthand for `{}::new().call(&ctx", name)?;
    for arg in &escaped_args {
        write!(w, ", &{}", arg.name)?;
    }
    writeln!(w, ")`.")?;
    writeln!(w, "///")?;
    writeln!(
        w,
        "/// See : <https://www.tensorflow.org/api_docs/python/tf/raw_ops/{}>",
        name
    )?;
    write!(w, "pub fn {}<'a", fn_name)?;
    for i in 0..escaped_args.len() {
        write!(w, ", T{}: crate::eager::ToTensorHandle<'a>", i)?;
    }
    write!(w, ">(ctx: &'a crate::eager::Context")?;
    for (i, arg) in escaped_args.iter().enumerate() {
        if !arg.has_number_attr() {
            write!(w, ", {}: &T{}", arg.name, i)?;
        } else {
            write!(w, ", {}: &[&T{}]", arg.name, i)?;
        };
    }
    write!(w, ") -> crate::Result<{}>", return_type(output_args.len()))?;
    writeln!(w, "{{")?;
    writeln!(w, "    let op = {}::new();", name)?;
    write!(w, "    op.call(ctx")?;
    for arg in escaped_args {
        write!(w, ", {}", arg.name)?;
    }
    writeln!(w, ")")?;
    writeln!(w, "}}")?;
    writeln!(w)?;
    Ok(())
}

fn write_attr_setter<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    writeln!(w)?;
    writeln!(w, "    /// Sets the `{}` attribute.", &attr.c_name)?;
    let rust_name = &attr.rust_name;
    let attr_type = &attr.attr_type;
    let mut value = "value.into()".to_string();
    if attr_type == "crate::Tensor" {
        value = format!(
            "::std::boxed::Box::new({}) as ::std::boxed::Box<dyn crate::AnyTensor>",
            value
        );
        writeln!(
            w,
            "    pub fn {}<T: crate::TensorType, ArgType: ::std::convert::Into<crate::Tensor<T>>>(mut self, value: ArgType) -> Self {{",
            rust_name
        )?;
    } else {
        writeln!(
            w,
            "    pub fn {}<ArgType: ::std::convert::Into<{}>>(mut self, value: ArgType) -> Self {{",
            rust_name, attr_type
        )?;
    }
    writeln!(
        w,
        "        self.{} = ::std::option::Option::Some({});",
        rust_name, value
    )?;
    writeln!(w, "        self")?;
    writeln!(w, "    }}")?;
    Ok(())
}

fn return_type(num_outputs: usize) -> String {
    match num_outputs {
        0 => "()".to_string(),
        1 => "crate::eager::TensorHandle<'a>".to_string(),
        n => format!("[crate::eager::TensorHandle<'a>; {}]", n),
    }
}

fn write_call_fn<W: Write>(
    w: &mut W,
    name: &str,
    fn_name: &str,
    input_args: &[&OpDef_ArgDef],
    output_args: &[String],
    attrs: &[Attr],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<InputArg> = input_args
        .iter()
        .map(|arg| InputArg {
            name: escaper.escape(&arg.name),
            number_attr: arg.number_attr.clone(),
            _type_attr: arg.type_attr.clone(),
        })
        .collect();
    writeln!(w, "    /// Execute {}.", fn_name)?;
    write!(w, "    pub fn call<'a")?;
    for i in 0..escaped_args.len() {
        write!(w, ", T{}: crate::eager::ToTensorHandle<'a>", i)?;
    }
    write!(w, ">(&self, ctx: &'a crate::eager::Context, ")?;
    let mut args_list = Vec::new();
    for (i, arg) in escaped_args.iter().enumerate() {
        let arg_str = if !arg.has_number_attr() {
            format!("{}: &T{}", arg.name, i)
        } else {
            format!("{}: &[&T{}]", arg.name, i)
        };
        args_list.push(arg_str);
    }
    let joined = args_list.join(", ");
    write!(w, "{}", joined)?;
    write!(w, ") -> crate::Result<{}>", return_type(output_args.len()))?;
    writeln!(w, "{{")?;
    writeln!(w, "    // Define Op")?;
    writeln!(w, "    let mut op = super::Op::new(ctx, \"{}\")?;", name)?;
    writeln!(w)?;
    writeln!(w, "    // Required input arguments")?;
    let mut number_attrs = HashSet::new();
    for arg in escaped_args {
        if !arg.has_number_attr() {
            writeln!(w, "    op.add_input(&{}.to_handle(ctx)?)?;", arg.name)?;
        } else {
            let arg_list = format!("{}_list", arg.name);
            writeln!(
                w,
                "    let mut {arg_list} = Vec::new();\n
    for t in {name} {{
       {arg_list}.push(t.to_handle(ctx)?);
    }}\n
    op.add_input_list(&{arg_list})?;",
                name = arg.name,
                arg_list = arg_list
            )?;
            number_attrs.insert(arg.number_attr.clone());
        };
    }
    writeln!(w)?;
    writeln!(w, "    // Attributes")?;
    for attr in attrs {
        if number_attrs.contains(&attr.c_name) {
            continue;
        }
        write_set_attr(w, attr)?;
    }
    writeln!(w)?;

    writeln!(
        w,
        "    // Set the device name where this Op will be executed"
    )?;
    writeln!(
        w,
        "    if let ::std::option::Option::Some(value) = &self.target_device_name {{"
    )?;
    writeln!(w, "        op.set_device(value)?;")?;
    writeln!(w, "    }}")?;
    writeln!(w, "    // Execute Op")?;

    match output_args.len() {
        0 => {
            writeln!(w, "    let _ = op.execute::<0>(ctx)?;")?;
            writeln!(w, "    Ok(())")?;
        }
        1 => {
            writeln!(w, "    let [h] = op.execute::<1>(ctx)?;")?;
            writeln!(w, "    Ok(h)")?;
        }
        n => {
            writeln!(w, "    let handles = op.execute::<{}>(ctx)?;", n)?;
            writeln!(w, "    Ok(handles)")?;
        }
    }
    writeln!(w, "}}")?;
    writeln!(w)?;
    Ok(())
}

fn write_attr<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    if attr.attr_type == "crate::Tensor" {
        writeln!(
            w,
            "    {}: ::std::option::Option<::std::boxed::Box<dyn crate::AnyTensor>>,",
            attr.rust_name
        )?;
    } else {
        writeln!(
            w,
            "    {}: ::std::option::Option<{}>,",
            attr.rust_name, attr.attr_type
        )?;
    }
    Ok(())
}

fn define_op<W: Write>(
    w: &mut W,
    keywords: &HashSet<String>,
    fn_escaper: &mut Escaper,
    struct_escaper: &mut Escaper,
    op: &OpDef,
) -> Result<(), io::Error> {
    let fn_name = fn_escaper.escape(&snake_name(&op.name));
    let name = struct_escaper.escape(&op.name);
    let op_name = op.name.clone();
    let input_args: Vec<_> = op.input_arg.iter().collect();

    // c_name that has number_attr
    let arg_with_number_attr: HashSet<String> = op
        .input_arg
        .iter()
        .filter(|arg| !arg.number_attr.is_empty())
        .map(|arg| arg.name.clone())
        .collect();
    let skip_attrs = {
        let input_number_attrs: HashSet<String> = op
            .input_arg
            .iter()
            .filter(|arg| !arg.number_attr.is_empty())
            .map(|arg| arg.number_attr.clone())
            .collect();
        // Collect type attributes that do not affect execution results.
        // This type attr is useful when we make a better documentation for each Op.
        let type_attrs: HashSet<String> = op
            .input_arg
            .iter()
            .filter(|arg| !arg.type_attr.is_empty())
            .map(|arg| arg.type_attr.clone())
            .collect();
        let type_list_attrs: HashSet<String> = op
            .input_arg
            .iter()
            .filter(|arg| !arg.type_list_attr.is_empty())
            .map(|arg| arg.type_list_attr.clone())
            .collect();

        let mut skip_attrs = HashSet::<String>::new();
        skip_attrs.extend(input_number_attrs);
        skip_attrs.extend(type_attrs);
        skip_attrs.extend(type_list_attrs);
        skip_attrs
    };

    let output_args: Vec<_> = op.output_arg.iter().map(|arg| arg.name.clone()).collect();
    let mut attrs = Vec::new();
    let mut attr_escaper = Escaper::new(keywords);
    for attr in op.attr.iter() {
        // skip if the attr is for type annotation
        if skip_attrs.contains(&attr.get_name().to_string()) {
            continue;
        }

        let rust_type = match &attr.field_type as &str {
            // See OpDef.AttrDef.type in $TENSORFLOW/tensorflow/core/framework/op_def.proto
            // and AttrValue in $TENSORFLOW/tensorflow/core/framework/attr_value.proto
            "string" => "::std::string::String",
            "int" => "i64",
            "float" => "f32",
            "bool" => "bool",
            "type" => "crate::DataType",
            "shape" => "crate::Shape",
            "tensor" => "crate::Tensor",
            "func" => "::std::string::String",
            "list(string)" => "::std::vec::Vec<::std::string::String>",
            "list(int)" => "::std::vec::Vec<i64>",
            "list(float)" => "::std::vec::Vec<f32>",
            "list(bool)" => "::std::vec::Vec<bool>",
            "list(type)" => "::std::vec::Vec<crate::DataType>",
            "list(shape)" => "::std::vec::Vec<crate::Shape>",
            "list(tensor)" => "::std::vec::Vec<crate::Tensor>",
            "list(func)" => "::std::vec::Vec<::std::string::String>",
            t => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "unrecognized field type {:?} for attribute {:?} of op {:?}",
                        t, &attr.name, &op_name
                    ),
                ))
            }
        };
        let default_value = match attr.default_value.as_ref() {
            Some(t) => t.value.to_owned(),
            _ => None,
        };
        attrs.push(Attr {
            rust_name: attr_escaper.escape(&attr.name),
            attr_type: rust_type.to_string(),
            c_name: attr.name.clone(),
            default_value,
        });
    }
    writeln!(w, "/// {} ", name)?;
    writeln!(w, "///")?;
    writeln!(
        w,
        "/// See : <https://www.tensorflow.org/api_docs/python/tf/raw_ops/{}>",
        name
    )?;
    let target_device_name_attr = Attr {
        rust_name: "target_device_name".into(),
        attr_type: "::std::string::String".into(),
        c_name: "".into(),
        default_value: None,
    };
    // derive clone if possible
    if attrs.iter().any(|attr| attr.attr_type == "crate::Tensor") {
        writeln!(w, "#[derive(::std::fmt::Debug)]")?;
    } else {
        writeln!(w, "#[derive(::std::fmt::Debug, ::std::clone::Clone)]")?;
    }
    writeln!(w, "pub struct {} {{", name)?;
    for attr in &attrs {
        if arg_with_number_attr.contains(&attr.c_name) {
            continue;
        }
        write_attr(w, attr)?;
    }
    writeln!(
        w,
        "    /// (Rust wrapper specific) A device name where this op will be executed"
    )?;
    write_attr(w, &target_device_name_attr)?;
    writeln!(w, "}}")?;

    writeln!(w, "impl ::std::default::Default for {} {{", name)?;
    writeln!(w, "    fn default() -> Self {{")?;
    writeln!(w, "        Self {{")?;
    for attr in &attrs {
        if arg_with_number_attr.contains(&attr.c_name) {
            continue;
        }
        write!(w, "            {}: ", attr.rust_name)?;
        match &attr.default_value {
            Some(AttrValue_oneof_value::s(s)) => {
                if s.is_empty() {
                    write!(w, "None")?;
                } else {
                    let msg = std::str::from_utf8(s).expect("Failed to decode as utf-8");
                    write!(w, "Some(::std::string::String::from(\"{}\"))", msg)?;
                }
            }
            Some(AttrValue_oneof_value::i(i)) => write!(w, "Some({}i64)", i)?,
            Some(AttrValue_oneof_value::f(f)) => {
                if f == &f32::INFINITY {
                    write!(w, "Some(f32::INFINITY)")?;
                } else if f == &f32::NEG_INFINITY {
                    write!(w, "Some(f32::NEG_INFINITY)")?;
                } else if f == &f32::NAN {
                    write!(w, "Some(f32::NAN)")?;
                } else {
                    write!(w, "Some({}f32)", f)?;
                }
            }
            Some(AttrValue_oneof_value::b(b)) => write!(w, "Some({})", b)?,
            Some(AttrValue_oneof_value::field_type(t)) => match t.descriptor().name() {
                "DT_FLOAT" => write!(w, "Some(crate::DataType::Float)")?,
                "DT_DOUBLE" => write!(w, "Some(crate::DataType::Double)")?,
                "DT_INT32" => write!(w, "Some(crate::DataType::Int32)")?,
                "DT_UINT8" => write!(w, "Some(crate::DataType::UInt8)")?,
                "DT_INT16" => write!(w, "Some(crate::DataType::Int16)")?,
                "DT_INT8" => write!(w, "Some(crate::DataType::Int8)")?,
                "DT_STRING" => write!(w, "Some(crate::DataType::String)")?,
                "DT_COMPLEX64" => write!(w, "Some(crate::DataType::Complex64)")?,
                "DT_INT64" => write!(w, "Some(crate::DataType::Int64)")?,
                "DT_BOOL" => write!(w, "Some(crate::DataType::Bool)")?,
                "DT_QINT8" => write!(w, "Some(crate::DataType::QInt8)")?,
                "DT_QUINT8" => write!(w, "Some(crate::DataType::QUInt8)")?,
                "DT_QINT32" => write!(w, "Some(crate::DataType::QInt32)")?,
                "DT_BFLOAT16" => write!(w, "Some(crate::DataType::BFloat16)")?,
                "DT_QINT16" => write!(w, "Some(crate::DataType::QInt16)")?,
                "DT_QUINT16" => write!(w, "Some(crate::DataType::QUInt16)")?,
                "DT_UINT16" => write!(w, "Some(crate::DataType::UInt16)")?,
                "DT_COMPLEX128" => write!(w, "Some(crate::DataType::Complex128)")?,
                "DT_HALF" => write!(w, "Some(crate::DataType::Half)")?,
                "DT_UINT32" => write!(w, "Some(crate::DataType::UInt32)")?,
                "DT_UINT64" => write!(w, "Some(crate::DataType::UInt64)")?,
                _ => panic!("{} is not supported", t.descriptor().name()),
            },
            Some(AttrValue_oneof_value::shape(shape)) => {
                let dims: Vec<_> = shape
                    .get_dim()
                    .iter()
                    .map(|x| format!("{}", x.get_size()))
                    .collect();
                if dims.is_empty() {
                    write!(w, "None")?;
                } else {
                    write!(w, "Some(crate::Shape::from(&[{}])", dims.join(", "))?;
                }
            }
            Some(AttrValue_oneof_value::tensor(_tensor)) => {
                write!(w, "None")?;
                eprintln!("default value for tensor is not supported")
            }
            Some(AttrValue_oneof_value::list(list)) => {
                match attr.attr_type.as_str() {
                    "::std::vec::Vec<i64>" => {
                        write!(w, "Some(vec!{:?})", list.i)?;
                    }
                    "::std::vec::Vec<f32>" => {
                        write!(w, "Some(vec!{:?})", list.f)?;
                    }
                    "::std::vec::Vec<bool>" => {
                        write!(w, "Some(vec!{:?})", list.b)?;
                    }
                    "::std::vec::Vec<::std::string::String>" => {
                        let msgs: Vec<_> = list
                            .s
                            .iter()
                            .map(|buf| std::str::from_utf8(buf).expect("Failed to decode as utf-8"))
                            .collect();
                        write!(w, "Some(vec![{}])", msgs.join(", "))?;
                    }
                    "::std::vec::Vec<crate::Shape>" => {
                        if list.shape.is_empty() {
                            write!(w, "Some(vec![])")?;
                        } else {
                            eprintln!("default value for {} is not supported.", attr.attr_type);
                            write!(w, "None")?;
                        }
                    }
                    "::std::vec::Vec<crate::DataType>" => {
                        if list.field_type.is_empty() {
                            write!(w, "Some(vec![])")?;
                        } else {
                            eprintln!("default value for {} is not supported.", attr.attr_type);
                            write!(w, "None")?;
                        }
                    }
                    _ => {
                        eprintln!("default value for {} is not supported.", attr.attr_type);
                        write!(w, "None")?;
                    }
                };
            }
            Some(AttrValue_oneof_value::func(_func)) => {
                eprintln!("default value for func is not supported");
                write!(w, "None")?;
            }
            Some(AttrValue_oneof_value::placeholder(_placeholder)) => {
                eprintln!("default value for placeholder is not supported");
                write!(w, "None")?;
            }
            _ => write!(w, "None")?,
        }
        writeln!(w, ",")?;
    }
    writeln!(
        w,
        "            {}: None,",
        target_device_name_attr.rust_name
    )?;
    writeln!(w, "        }}")?;
    writeln!(w, "    }}")?;
    writeln!(w, "}}")?;

    writeln!(
        w,
        r#"impl {name} {{
    /// Creates a new `{name}`.
    pub fn new() -> Self {{
        Self::default()
    }}"#,
        name = name
    )?;
    for attr in &attrs {
        if arg_with_number_attr.contains(&attr.c_name) {
            continue;
        }
        write_attr_setter(w, attr)?;
    }
    write_attr_setter(w, &target_device_name_attr)?;
    writeln!(w)?;
    write_call_fn(
        w,
        &name,
        &fn_name,
        &input_args,
        &output_args,
        &attrs,
        keywords,
    )?;
    writeln!(w, "}}")?;
    writeln!(w)?;
    write_short_fn(w, &name, &fn_name, &input_args, &output_args, keywords)?;
    Ok(())
}

fn snake_name(name: &str) -> String {
    let mut s = String::new();
    let mut was_lower = false;
    for c in name.chars() {
        if c.is_uppercase() {
            if was_lower {
                s.push('_');
            }
            was_lower = false;
        } else {
            was_lower = true;
        }
        for cc in c.to_lowercase() {
            s.push(cc);
        }
    }
    s
}

struct Escaper<'a> {
    keywords: &'a HashSet<String>,
    used_names: HashSet<String>,
}

impl<'a> Escaper<'a> {
    fn new(keywords: &'a HashSet<String>) -> Self {
        Self {
            keywords,
            used_names: HashSet::new(),
        }
    }

    fn escape(&mut self, name: &str) -> String {
        let suffix = if self.keywords.contains(name) {
            "_"
        } else {
            ""
        };
        let mut candidate = format!("{}{}", name, suffix);
        let mut i = 2;
        while self.used_names.contains(&candidate) {
            candidate = format!("{}_{}", name, i);
            i += 1;
        }
        self.used_names.insert(candidate.clone());
        candidate
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let tensorflow_folder = Path::new(&args[1]);
    let output_folder = Path::new(&args[2]);
    let ops_bytes = fs::read(
        tensorflow_folder
            .join("tensorflow/core/ops/ops.pbtxt")
            .to_str()
            .ok_or("Unable to format path for tensorflow folder")?,
    )?;
    let ops = parser::parse(&ops_bytes).map_err(|e| {
        println!("Parse error at {:?}", e.pos);
        if let Some(p) = &e.pos {
            let input = String::from_utf8_lossy(&ops_bytes);
            println!("Previous: {}", &input[0..*p]);
            println!("Next: {}", &input[*p..]);
        }
        e
    })?;
    let keywords: HashSet<String> = [
        "abstract",
        "as",
        "async",
        "await",
        "become",
        "box",
        "break",
        "const",
        "continue",
        "crate",
        "do",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "final",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "macro",
        "match",
        "mod",
        "move",
        "mut",
        "override",
        "priv",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "try",
        "type",
        "typeof",
        "unsafe",
        "unsized",
        "use",
        "virtual",
        "where",
        "while",
        "yield",
        // These aren't technically keywords, but there doesn't appear to be a
        // way to refer to these types (e.g. qualified type names) if the name
        // has been shadowed by something else, so we treat them as keywords.
        "bool",
        "char",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "str",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        // new and call aren't keywords, but they still can't be used because they
        // would clash with methods we're providing.
        "new",
        "call",
        "ctx",
        "target_device_name",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut out = BufWriter::new(File::create(output_folder.join("src/eager/op/raw_ops.rs"))?);
    write!(
        &mut out,
        r#"// DO NOT EDIT. Generated by tensorflow-op-codegen/src/main.rs.

"#
    )?;
    let mut fn_escaper = Escaper::new(&keywords);
    let mut struct_escaper = Escaper::new(&keywords);
    for op in ops {
        define_op(
            &mut out,
            &keywords,
            &mut fn_escaper,
            &mut struct_escaper,
            &op,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_name() {
        assert_eq!(&snake_name("foo"), "foo");
        assert_eq!(&snake_name("fooBar"), "foo_bar");
        assert_eq!(&snake_name("FooBar"), "foo_bar");
        assert_eq!(&snake_name("abcXYZ"), "abc_xyz");
        assert_eq!(&snake_name("abcXYZdef"), "abc_xyzdef");
    }

    #[test]
    fn test_escaper() {
        let mut keywords = HashSet::new();
        keywords.insert("fn".to_string());
        let mut escaper = Escaper::new(&keywords);
        assert_eq!(&escaper.escape("fn"), "fn_");
        assert_eq!(&escaper.escape("fn"), "fn_2");
        assert_eq!(&escaper.escape("fn"), "fn_3");
        assert_eq!(&escaper.escape("foo"), "foo");
        assert_eq!(&escaper.escape("foo"), "foo_2");
    }
}
