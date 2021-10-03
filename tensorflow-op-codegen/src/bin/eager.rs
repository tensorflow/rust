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

#[derive(Clone)]
struct Attr {
    rust_name: String,
    attr_type: String,
    c_name: String,
    default_value: Option<AttrValue_oneof_value>,
}

fn write_set_attr<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    let c_name = &attr.c_name;
    let rust_name = &attr.rust_name;
    let setter = match attr.attr_type.as_str() {
        "::std::string::String" => format!("op.set_attr_string(\"{}\", value)", c_name),
        "crate::DataType" => format!("op.set_attr_type(\"{}\", *value)", c_name),
        "bool" => format!("op.set_attr_bool(\"{}\", *value)", c_name),
        "f32" => format!("op.set_attr_float(\"{}\", *value)", c_name),
        "i64" => format!("op.set_attr_int(\"{}\", *value)", c_name),
        "crate::Shape" => format!("op.set_attr_shape(\"{}\", value)?", c_name),
        "crate::Tensor" => format!("op.set_attr_any_tensor(\"{}\", value)?", c_name),
        "::std::vec::Vec<::std::string::String>" => {
            format!("op.set_attr_string_list(\"{}\", value)", c_name)
        }
        "::std::vec::Vec<f32>" => {
            format!("op.set_attr_float_list(\"{}\", value)", c_name)
        }
        "::std::vec::Vec<i64>" => format!("op.set_attr_int_list(\"{}\", value)", c_name),
        "::std::vec::Vec<crate::DataType>" => {
            format!("op.set_attr_type_list(\"{}\", value)", c_name)
        }
        "::std::vec::Vec<crate::Shape>" => {
            format!("op.set_attr_shape_list(\"{}\", value)?", c_name)
        }
        ty => panic!("Unrecognized attribute type for {}: {}", attr.rust_name, ty),
    };
    write!(
        w,
        "    if let ::std::option::Option::Some(value) = &self.{} {{\n",
        rust_name
    )?;
    write!(w, "        {};\n", setter)?;
    write!(w, "    }}\n")?;
    Ok(())
}

fn write_short_fn<W: Write>(
    w: &mut W,
    name: &str,
    fn_name: &str,
    input_args: &[String],
    output_args: &[String],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<_> = input_args.iter().map(|arg| escaper.escape(&arg)).collect();
    write!(w, "/// {} with default options.\n", fn_name)?;
    write!(w, "pub fn {}<'a", fn_name)?;
    for i in 0..escaped_args.len() {
        write!(w, ", T{}: crate::eager::ToHandle<'a>", i)?;
    }
    write!(w, ">(ctx: &'a crate::eager::Context")?;
    for (i, arg) in escaped_args.iter().enumerate() {
        write!(w, ", {}: T{}", arg, i)?;
    }
    write!(w, ") -> crate::Result<")?;
    match output_args.len() {
        0 => write!(w, "()")?,
        1 => write!(w, "crate::eager::TensorHandle")?,
        n => write!(w, "[crate::eager::TensorHandle; {}]", n)?,
    };
    write!(w, ">\n")?;
    write!(w, "{{\n")?;
    write!(w, "    let op = {}::new();\n", name)?;
    write!(w, "    op.call(ctx")?;
    for arg in escaped_args {
        write!(w, ", {}", arg)?;
    }
    write!(w, ")\n")?;
    write!(w, "}}\n\n")?;
    Ok(())
}

fn write_attr_setter<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    write!(w, "\n")?;
    write!(w, "    /// Sets the `{}` attribute.\n", &attr.c_name)?;
    let rust_name = &attr.rust_name;
    let attr_type = &attr.attr_type;
    let mut value = "value.into()".to_string();
    if attr_type == "crate::Tensor" {
        value = format!(
            "(::std::boxed::Box::new({}) as ::std::boxed::Box<dyn crate::AnyTensor>)",
            value
        );
        write!(
            w,
            "    pub fn {}<T: crate::TensorType, ArgType: ::std::convert::Into<crate::Tensor<T>>>(mut self, value: ArgType) -> Self {{\n",
            rust_name
        )?;
    } else {
        write!(
            w,
            "    pub fn {}<ArgType: ::std::convert::Into<{}>>(mut self, value: ArgType) -> Self {{\n",
            rust_name, attr_type
        )?;
    }
    write!(
        w,
        "        self.{} = ::std::option::Option::Some({});\n",
        rust_name, value
    )?;
    write!(w, "        self\n")?;
    write!(w, "    }}\n")?;
    Ok(())
}

fn write_call_fn<W: Write>(
    w: &mut W,
    name: &str,
    fn_name: &str,
    input_args: &[String],
    output_args: &[String],
    attrs: &[Attr],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<_> = input_args.iter().map(|arg| escaper.escape(&arg)).collect();
    write!(w, "/// Execute {}.\n", fn_name)?;
    write!(w, "    pub fn call<'a")?;
    for i in 0..escaped_args.len() {
        write!(w, ", T{}: crate::eager::ToHandle<'a>", i)?;
    }
    write!(w, ">(&self, ctx: &'a crate::eager::Context, ")?;
    let args_list: Vec<_> = escaped_args
        .iter()
        .enumerate()
        .map(|(i, arg)| format!("{}: T{}", arg, i))
        .collect();
    let joined = args_list.join(", ");
    write!(w, "{}", joined)?;
    write!(w, ") -> crate::Result<")?;
    match output_args.len() {
        0 => write!(w, "()")?,
        1 => write!(w, "crate::eager::TensorHandle<'a>")?,
        n => write!(w, "[crate::eager::TensorHandle<'a>; {}]", n)?,
    };
    write!(w, ">\n")?;
    write!(w, "{{\n")?;
    write!(w, "    let status = crate::Status::new();\n")?;
    write!(w, "\n")?;
    write!(w, "    // Define Op\n")?;
    let op_mut = if escaped_args.len() + attrs.len() > 0 {
        "mut"
    } else {
        ""
    };
    write!(
        w,
        "    let {} op = crate::eager::Op::new(&ctx, \"{}\")?;\n",
        op_mut, name
    )?;
    write!(w, "\n")?;
    write!(w, "    // Required input arguments\n")?;
    for arg in escaped_args {
        write!(w, "    op.add_input(&{}.to_handle(&ctx)?)?;\n", arg)?;
    }
    write!(w, "\n")?;
    write!(w, "    // Attributes\n")?;
    for attr in attrs {
        write_set_attr(w, attr)?;
    }
    write!(w, "\n")?;
    write!(w, "    // Execute Op\n")?;
    write!(w, "    let mut num_output = {};\n", output_args.len())?;
    write!(
        w,
        "    let mut res = [std::ptr::null_mut::<tensorflow_sys::TFE_TensorHandle>(); {}];\n",
        output_args.len()
    )?;
    write!(w, "    unsafe {{\n")?;
    write!(w, "        tensorflow_sys::TFE_Execute(op.inner, res.as_mut_ptr(), (&mut num_output) as *mut i32, status.inner);\n")?;
    write!(w, "    }};\n")?;
    write!(w, "    if status.is_ok() {{\n")?;
    match output_args.len() {
        0 => {
            write!(w, "        return Ok(());\n")?;
        }
        1 => {
            write!(w, "        let ret = unsafe {{ \n")?;
            write!(
                w,
                "            crate::eager::TensorHandle::from_tensor_handle(&ctx, res[0])\n",
            )?;
            write!(w, "        }};\n")?;
            write!(w, "        return Ok(ret);\n")?;
        }
        n => {
            write!(w, "        let ret = unsafe {{ [\n")?;
            for i in 0..n {
                write!(
                    w,
                    "            crate::eager::TensorHandle::from_tensor_handle(&ctx, res[{}]),\n",
                    i
                )?;
            }
            write!(w, "        ] }};\n")?;
            write!(w, "        return Ok(ret);\n")?;
        }
    };

    write!(w, "    }}\n")?;
    write!(w, "    Err(status)\n")?;
    write!(w, "}}\n\n")?;
    Ok(())
}

fn write_attr<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    if attr.attr_type == "crate::Tensor" {
        write!(
            w,
            "    {}: ::std::option::Option<::std::boxed::Box<dyn crate::AnyTensor>>,\n",
            attr.rust_name
        )?;
    } else {
        write!(
            w,
            "    {}: ::std::option::Option<{}>,\n",
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
    let input_args: Vec<_> = op.input_arg.iter().map(|arg| arg.name.clone()).collect();
    let output_args: Vec<_> = op.output_arg.iter().map(|arg| arg.name.clone()).collect();
    let mut attrs = Vec::new();
    let mut attr_escaper = Escaper::new(keywords);
    for attr in op.attr.iter() {
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
    write!(w, "/// {} \n", name)?;
    write!(w, "#[derive(::std::fmt::Debug)]\n")?;
    write!(w, "pub struct {} {{\n", name)?;
    for attr in &attrs {
        write_attr(w, &attr)?;
    }
    write!(w, "}}\n")?;

    write!(w, "impl ::std::default::Default for {} {{\n", name)?;
    write!(w, "    fn default() -> Self {{\n")?;
    write!(w, "        Self {{\n")?;
    for attr in &attrs {
        write!(w, "            {}: ", attr.rust_name)?;
        match &attr.default_value {
            Some(AttrValue_oneof_value::s(s)) => {
                if s.len() == 0 {
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
                "DT_HAFL" => write!(w, "Some(crate::DataType::Hafl)")?,
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
                if dims.len() == 0 {
                    write!(w, "None")?;
                } else {
                    write!(w, "Some(crate::Shape::from(&[{}])", dims.join(", "))?;
                }
            }
            Some(AttrValue_oneof_value::tensor(tensor)) => {
                dbg!(tensor);
                write!(w, "None")?;
                eprintln!("tensor is not supported")
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
                        dbg!(&list.shape);
                        eprintln!("{} is not supported.", attr.attr_type);
                        write!(w, "None")?;
                    }
                    "::std::vec::Vec<crate::DataType>" => {
                        dbg!(&list.field_type);
                        eprintln!("{} is not supported.", attr.attr_type);
                        write!(w, "None")?;
                    }
                    _ => {
                        eprintln!("{} is not supported.", attr.attr_type);
                        write!(w, "None")?;
                    }
                };
            }
            Some(AttrValue_oneof_value::func(func)) => {
                dbg!(func);
                eprintln!("func is not supported");
                write!(w, "None")?;
            }
            Some(AttrValue_oneof_value::placeholder(placeholder)) => {
                dbg!(placeholder);
                eprintln!("placeholder is not supported");
                write!(w, "None")?;
            }
            _ => write!(w, "None")?,
        }
        write!(w, ",\n")?;
    }
    write!(w, "        }}\n")?;
    write!(w, "    }}\n")?;
    write!(w, "}}\n")?;

    write!(
        w,
        r#"impl {name} {{
    /// Creates a new `{name}`.
    pub fn new() -> Self {{
        Self::default()
    }}
"#,
        name = name
    )?;
    for attr in &attrs {
        write_attr_setter(w, attr)?;
    }
    write!(w, "\n")?;
    write_call_fn(
        w,
        &name,
        &fn_name,
        &input_args,
        &output_args,
        &attrs,
        &keywords,
    )?;
    write!(w, "}}\n")?;
    write!(w, "\n")?;
    write_short_fn(w, &name, &fn_name, &input_args, &output_args, &keywords)?;
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
        "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
        "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "if", "impl", "in",
        "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "try", "type",
        "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
        // These aren't technically keywords, but there doesn't appear to be a
        // way to refer to these types (e.g. qualified type names) if the name
        // has been shadowed by something else, so we treat them as keywords.
        "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str", "u8",
        "u16", "u32", "u64", "u128", "usize",
        // new and call aren't keywords, but they still can't be used because they would clash
        // with methods we're providing.
        "new", "call",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut out = BufWriter::new(File::create(output_folder.join("src/eager/raw_ops.rs"))?);
    write!(
        &mut out,
        "{}",
        r#"// DO NOT EDIT. Generated by tensorflow-op-codegen/src/main.rs.
#![allow(
    missing_copy_implementations,
    missing_docs,
    non_snake_case,
    trivial_casts,
    unused_parens,
    unused_qualifications
)]
use tensorflow_sys;
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
