// This parses the ops.pbtxt file in TensorFlow and returns a list of op definitions. This is
// currently implemented directly with nom to parse the text proto, but ideally this would use a
// proto library with support for parsing text protos.

use crate::protos::AttrValue;
use crate::protos::AttrValue_ListValue;
use crate::protos::DataType;
use crate::protos::FullTypeDef;
use crate::protos::FullTypeId;
use crate::protos::OpDef;
use crate::protos::OpDef_ArgDef;
use crate::protos::OpDef_AttrDef;
use crate::protos::OpDeprecation;
use crate::protos::TensorProto;
use crate::protos::TensorShapeProto;
use crate::protos::TensorShapeProto_Dim;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::anychar;
use nom::character::complete::char;
use nom::character::complete::none_of;
use nom::character::complete::one_of;
use nom::combinator::cut;
use nom::combinator::map;
use nom::combinator::map_res;
use nom::error::context;
use nom::error::make_error;
use nom::error::ErrorKind;
use nom::error::VerboseError;
use nom::multi::many0;
use nom::multi::many1;
use nom::multi::many_till;
use nom::multi::separated_list0;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::terminated;
use nom::IResult;
use protobuf::Message;
use protobuf::ProtobufEnum;
use protobuf::ProtobufError;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::result::Result;

fn merge_protos<M: Message>(ops: Vec<M>) -> Result<M, ProtobufError> {
    let mut out = Vec::new();
    for op in ops {
        op.write_to_vec(&mut out)?;
    }
    Message::parse_from_bytes(&out)
}

type ParseResult<'a, O> = IResult<&'a [u8], O, VerboseError<&'a [u8]>>;

fn space<'a>() -> impl FnMut(&'a [u8]) -> ParseResult<'a, ()> {
    map(many0(one_of(" \t\r\n")), |_| ())
}

fn eof<'a>() -> impl Fn(&'a [u8]) -> ParseResult<'a, ()> {
    move |input: &[u8]| {
        if input.len() == 0 {
            Ok((input, ()))
        } else {
            Err(nom::Err::Error(make_error(input, ErrorKind::Eof)))
        }
    }
}

fn string<'a>(input: &'a [u8]) -> ParseResult<'a, String> {
    let string_char = alt((
        none_of("\"\\"),
        map(tag("\\n"), |_| '\n'),
        map(tag("\\t"), |_| '\t'),
        map(tag("\\\""), |_| '"'),
        map(tag("\\'"), |_| '\''),
    ));
    delimited(
        char('"'),
        map(many0(string_char), |v| v.into_iter().collect()),
        char('"'),
    )(input)
}

fn boolean<'a>(input: &'a [u8]) -> ParseResult<'a, bool> {
    alt((map(tag("true"), |_| true), map(tag("false"), |_| false)))(input)
}

fn int64<'a>(input: &'a [u8]) -> ParseResult<'a, i64> {
    map_res(many1(one_of("0123456789+-")), |v| {
        str::parse::<i64>(&v.iter().collect::<String>())
    })(input)
}

fn int32<'a>(input: &'a [u8]) -> ParseResult<'a, i32> {
    map_res(many1(one_of("0123456789+-")), |v| {
        str::parse::<i32>(&v.iter().collect::<String>())
    })(input)
}

fn float_<'a>(input: &'a [u8]) -> ParseResult<'a, f32> {
    map_res(many1(one_of("0123456789+-.aefinAEFIN")), |v| {
        str::parse::<f32>(&v.iter().collect::<String>())
    })(input)
}

fn identifier<'a>(input: &'a [u8]) -> ParseResult<'a, String> {
    let identifier_start = map_res(anychar, |c| match c {
        'A'..='Z' | 'a'..='z' | '_' => Ok(c),
        _ => Err(nom::Err::<VerboseError<&'a [u8]>>::Error(make_error(
            input,
            ErrorKind::OneOf,
        ))),
    });
    let identifier_part = map_res(anychar, |c| match c {
        'A'..='Z' | 'a'..='z' | '_' | '0'..='9' => Ok(c),
        _ => Err(nom::Err::<VerboseError<&'a [u8]>>::Error(make_error(
            input,
            ErrorKind::OneOf,
        ))),
    });
    map(
        pair(identifier_start, many0(identifier_part)),
        |(first, rest)| [first].iter().chain(rest.iter()).collect::<String>(),
    )(input)
}

fn data_type<'a>(input: &'a [u8]) -> ParseResult<'a, DataType> {
    map_res(identifier, |s| match &s as &str {
        "DT_BFLOAT16" => Ok(DataType::DT_BFLOAT16),
        "DT_HALF" => Ok(DataType::DT_HALF),
        "DT_FLOAT" => Ok(DataType::DT_FLOAT),
        "DT_DOUBLE" => Ok(DataType::DT_DOUBLE),
        "DT_COMPLEX64" => Ok(DataType::DT_COMPLEX64),
        "DT_COMPLEX128" => Ok(DataType::DT_COMPLEX128),
        "DT_INT8" => Ok(DataType::DT_INT8),
        "DT_INT16" => Ok(DataType::DT_INT16),
        "DT_INT32" => Ok(DataType::DT_INT32),
        "DT_INT64" => Ok(DataType::DT_INT64),
        "DT_UINT8" => Ok(DataType::DT_UINT8),
        "DT_UINT16" => Ok(DataType::DT_UINT16),
        "DT_UINT32" => Ok(DataType::DT_UINT32),
        "DT_UINT64" => Ok(DataType::DT_UINT64),
        "DT_QINT8" => Ok(DataType::DT_QINT8),
        "DT_QINT16" => Ok(DataType::DT_QINT16),
        "DT_QINT32" => Ok(DataType::DT_QINT32),
        "DT_QUINT8" => Ok(DataType::DT_QUINT8),
        "DT_QUINT16" => Ok(DataType::DT_QUINT16),
        "DT_STRING" => Ok(DataType::DT_STRING),
        "DT_VARIANT" => Ok(DataType::DT_VARIANT),
        "DT_BOOL" => Ok(DataType::DT_BOOL),
        "DT_RESOURCE" => Ok(DataType::DT_RESOURCE),
        _ => Err(nom::Err::<VerboseError<&'a [u8]>>::Error(make_error(
            input,
            ErrorKind::Alt,
        ))),
    })(input)
}

fn type_id<'a>(input: &'a [u8]) -> ParseResult<'a, FullTypeId> {
    map_res(identifier, |s| {
        match FullTypeId::values()
            .iter()
            .filter(|d| d.descriptor().name() == s)
            .next()
        {
            Some(d) => Ok(FullTypeId::from_i32(d.value()).unwrap()),
            None => Err(nom::Err::<VerboseError<&'a [u8]>>::Error(make_error(
                input,
                ErrorKind::Alt,
            ))),
        }
    })(input)
}

fn scalar_field<'a, M, T, Parser, ParserFactory, S>(
    name: &'a str,
    parser_factory: ParserFactory,
    setter: S,
) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    Parser: Fn(&'a [u8]) -> ParseResult<'a, T>,
    ParserFactory: Fn() -> Parser,
    S: Fn(&mut M, T),
{
    move |input: &'a [u8]| {
        let (input, _) = tag(name)(input)?;
        let (input, _) = char(':')(input)?;
        let (input, _) = space()(input)?;
        let (input, value) = cut(parser_factory())(input)?;
        let mut message = M::new();
        setter(&mut message, value);
        Ok((input, message))
    }
}

fn string_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, String),
{
    scalar_field(name, || string, setter)
}

fn int32_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, i32),
{
    scalar_field(name, || int32, setter)
}

fn int64_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, i64),
{
    scalar_field(name, || int64, setter)
}

fn float_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, f32),
{
    scalar_field(name, || float_, setter)
}

fn boolean_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, bool),
{
    scalar_field(name, || boolean, setter)
}

fn data_type_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, DataType),
{
    scalar_field(name, || data_type, setter)
}

fn type_id_field<'a, M, S>(name: &'a str, setter: S) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    S: Fn(&mut M, FullTypeId),
{
    scalar_field(name, || type_id, setter)
}

fn raw_message_field<'a, M, F>(name: &'a str, parser: F, input: &'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    F: Fn(&'a [u8]) -> ParseResult<'a, M>,
{
    let (input, _) = tag(name)(input)?;
    let (input, _) = space()(input)?;
    let (input, _) = char('{')(input)?;
    let (input, _) = space()(input)?;
    let (input, value) = cut(parser)(input)?;
    let (input, _) = space()(input)?;
    let (input, _) = char('}')(input)?;
    Ok((input, value))
}

fn message_field<'a, M, Submessage, Parser, ParserFactory, Setter>(
    name: &'a str,
    parser_factory: ParserFactory,
    setter: Setter,
) -> impl Fn(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    Submessage: Message,
    Parser: Fn(&'a [u8]) -> ParseResult<'a, Submessage>,
    ParserFactory: Fn() -> Parser,
    Setter: Fn(&mut M, Submessage),
{
    move |input: &'a [u8]| {
        let (input, value) = raw_message_field(name, parser_factory(), input)?;
        let mut op = M::new();
        setter(&mut op, value);
        Ok((input, op))
    }
}

fn message<'a, M, F>(field: F) -> impl FnMut(&'a [u8]) -> ParseResult<'a, M>
where
    M: Message,
    F: FnMut(&'a [u8]) -> ParseResult<'a, M>,
{
    map_res(separated_list0(space(), field), merge_protos)
}

fn tensor_shape_proto_dim<'a>(input: &'a [u8]) -> ParseResult<'a, TensorShapeProto_Dim> {
    message(alt((
        int64_field("size", TensorShapeProto_Dim::set_size),
        string_field("name", TensorShapeProto_Dim::set_name),
    )))(input)
}

fn tensor_shape_proto<'a>(input: &'a [u8]) -> ParseResult<'a, TensorShapeProto> {
    message(alt((
        boolean_field("unknown_rank", TensorShapeProto::set_unknown_rank),
        message_field(
            "dim",
            || tensor_shape_proto_dim,
            |m: &mut TensorShapeProto, v| m.mut_dim().push(v),
        ),
    )))(input)
}

fn tensor_proto<'a>(input: &'a [u8]) -> ParseResult<'a, TensorProto> {
    message(alt((
        data_type_field("dtype", TensorProto::set_dtype),
        message_field(
            "tensor_shape",
            || tensor_shape_proto,
            TensorProto::set_tensor_shape,
        ),
        int32_field("version_number", TensorProto::set_version_number),
        int32_field("int_val", |m: &mut TensorProto, v| m.mut_int_val().push(v)),
    )))(input)
}

fn attr_value_list_value<'a>(input: &'a [u8]) -> ParseResult<'a, AttrValue_ListValue> {
    message(alt((
        data_type_field("type", |m: &mut AttrValue_ListValue, v| {
            m.mut_field_type().push(v)
        }),
        string_field("s", |m: &mut AttrValue_ListValue, v| {
            m.mut_s().push(v.bytes().collect())
        }),
        int64_field("i", |m: &mut AttrValue_ListValue, v| m.mut_i().push(v)),
        float_field("f", |m: &mut AttrValue_ListValue, v| m.mut_f().push(v)),
    )))(input)
}

fn attr_value<'a>(input: &'a [u8]) -> ParseResult<'a, AttrValue> {
    message(alt((
        string_field("s", |m: &mut AttrValue, s| m.set_s(s.bytes().collect())),
        boolean_field("b", AttrValue::set_b),
        int64_field("i", AttrValue::set_i),
        float_field("f", AttrValue::set_f),
        string_field("s", |m: &mut AttrValue, v| m.set_s(v.bytes().collect())),
        message_field("list", || attr_value_list_value, AttrValue::set_list),
        message_field("tensor", || tensor_proto, AttrValue::set_tensor),
        message_field("shape", || tensor_shape_proto, AttrValue::set_shape),
        data_type_field("type", AttrValue::set_field_type),
    )))(input)
}

fn attr<'a>(input: &'a [u8]) -> ParseResult<'a, OpDef_AttrDef> {
    message(alt((
        string_field("name", OpDef_AttrDef::set_name),
        string_field("description", OpDef_AttrDef::set_description),
        string_field("type", OpDef_AttrDef::set_field_type),
        message_field(
            "default_value",
            || attr_value,
            OpDef_AttrDef::set_default_value,
        ),
        message_field(
            "allowed_values",
            || attr_value,
            OpDef_AttrDef::set_allowed_values,
        ),
        boolean_field("has_minimum", OpDef_AttrDef::set_has_minimum),
        int64_field("minimum", OpDef_AttrDef::set_minimum),
    )))(input)
}

fn experimental_full_type<'a>(input: &'a [u8]) -> ParseResult<'a, FullTypeDef> {
    message(alt((
        type_id_field("type_id", FullTypeDef::set_type_id),
        message_field(
            "args",
            || experimental_full_type,
            |m: &mut FullTypeDef, v| m.mut_args().push(v),
        ),
        string_field("s", FullTypeDef::set_s),
    )))(input)
}

fn arg_def<'a>(input: &'a [u8]) -> ParseResult<'a, OpDef_ArgDef> {
    message(alt((
        string_field("name", OpDef_ArgDef::set_name),
        string_field("description", OpDef_ArgDef::set_description),
        data_type_field("type", OpDef_ArgDef::set_field_type),
        string_field("type_attr", OpDef_ArgDef::set_type_attr),
        string_field("type_list_attr", OpDef_ArgDef::set_type_list_attr),
        string_field("number_attr", OpDef_ArgDef::set_number_attr),
        boolean_field("is_ref", OpDef_ArgDef::set_is_ref),
        message_field(
            "experimental_full_type",
            || experimental_full_type,
            |m: &mut OpDef_ArgDef, v| m.set_experimental_full_type(v),
        ),
    )))(input)
}

fn op_deprecation<'a>(input: &'a [u8]) -> ParseResult<'a, OpDeprecation> {
    message(alt((
        int32_field("version", OpDeprecation::set_version),
        string_field("explanation", OpDeprecation::set_explanation),
    )))(input)
}

fn op_def<'a>(input: &'a [u8]) -> ParseResult<'a, OpDef> {
    context(
        "OpDef",
        message(alt((
            string_field("name", OpDef::set_name),
            string_field("summary", OpDef::set_summary),
            string_field("description", OpDef::set_description),
            message_field("attr", || attr, |m: &mut OpDef, v| m.mut_attr().push(v)),
            message_field(
                "input_arg",
                || arg_def,
                |m: &mut OpDef, v| m.mut_input_arg().push(v),
            ),
            message_field(
                "output_arg",
                || arg_def,
                |m: &mut OpDef, v| m.mut_output_arg().push(v),
            ),
            boolean_field("is_aggregate", OpDef::set_is_aggregate),
            boolean_field("is_commutative", OpDef::set_is_commutative),
            boolean_field("is_stateful", OpDef::set_is_stateful),
            boolean_field(
                "is_distributed_communication",
                OpDef::set_is_distributed_communication,
            ),
            boolean_field(
                "allows_uninitialized_input",
                OpDef::set_allows_uninitialized_input,
            ),
            message_field("deprecation", || op_deprecation, OpDef::set_deprecation),
        ))),
    )(input)
}

fn op_def_array<'a>(input: &'a [u8]) -> ParseResult<'a, Vec<OpDef>> {
    terminated(
        preceded(
            space(),
            map(
                many_till(
                    terminated(|input| raw_message_field("op", op_def, input), space()),
                    eof(),
                ),
                |(ops, _)| ops,
            ),
        ),
        eof(),
    )(input)
}

pub fn parse(input: &[u8]) -> Result<Vec<OpDef>, ParseError> {
    match op_def_array(&input) {
        Ok(x) => Ok(x.1),
        Err(e) => {
            let pos = match &e {
                nom::Err::Incomplete(_) => None,
                nom::Err::Error(e) => {
                    if e.errors.len() == 0 {
                        None
                    } else {
                        Some(input.len() - e.errors[0].0.len())
                    }
                }
                nom::Err::Failure(e) => {
                    if e.errors.len() == 0 {
                        None
                    } else {
                        Some(input.len() - e.errors[0].0.len())
                    }
                }
            };
            let msg = match &e {
                nom::Err::Incomplete(_) => "incomplete".to_string(),
                nom::Err::Error(e) => e
                    .errors
                    .iter()
                    .map(|e| format!("{:?}", e.1))
                    .collect::<Vec<_>>()
                    .join(", "),
                nom::Err::Failure(e) => e
                    .errors
                    .iter()
                    .map(|e| format!("{:?}", e.1))
                    .collect::<Vec<_>>()
                    .join(", "),
            };
            Err(ParseError { msg, pos })
        }
    }
}

#[derive(Debug)]
pub struct ParseError {
    pub msg: String,
    pub pos: Option<usize>,
}

impl Error for ParseError {}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {:?}", self.msg, self.pos)
    }
}
