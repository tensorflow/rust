#!/bin/sh

if uname -a | grep x86_64 >/dev/null ; then
    ARCH_SUFFIX=amd64
else
    ARCH_SUFFIX=i386
fi

# UJA is for update-java-alternatives
if [[ -d "/usr/lib/jvm/java-6-openjdk-$ARCH_SUFFIX" ]] ; then
    OPENJDK6_UJA_ALIAS="java-1.6.0-openjdk-$ARCH_SUFFIX"
    OPENJDK6_JAVA_HOME="/usr/lib/jvm/java-6-openjdk-$ARCH_SUFFIX"
else
    OPENJDK6_UJA_ALIAS="java-1.6.0-openjdk"
    OPENJDK6_JAVA_HOME="/usr/lib/jvm/java-6-openjdk"
fi

if [[ -d "/usr/lib/jvm/java-7-openjdk-$ARCH_SUFFIX" ]] ; then
    OPENJDK7_UJA_ALIAS="java-1.7.0-openjdk-$ARCH_SUFFIX"
    OPENJDK7_JAVA_HOME="/usr/lib/jvm/java-7-openjdk-$ARCH_SUFFIX"
else
    OPENJDK7_UJA_ALIAS="java-1.7.0-openjdk"
    OPENJDK7_JAVA_HOME="/usr/lib/jvm/java-7-openjdk"
fi

if [[ -d "/usr/lib/jvm/java-8-openjdk-$ARCH_SUFFIX" ]] ; then
    OPENJDK8_UJA_ALIAS="java-1.8.0-openjdk-$ARCH_SUFFIX"
    OPENJDK8_JAVA_HOME="/usr/lib/jvm/java-8-openjdk-$ARCH_SUFFIX"
else
    OPENJDK8_UJA_ALIAS="java-1.8.0-openjdk"
    OPENJDK8_JAVA_HOME="/usr/lib/jvm/java-8-openjdk"
fi

# The java::oraclejdk7 recipe from github.com/travis-ci/travis-cookbooks
# takes care of this alias. We take it for granted here. MK.
if [[ -d "/usr/lib/jvm/java-7-oracle-$ARCH_SUFFIX" ]] ; then
    ORACLEJDK7_UJA_ALIAS="java-7-oracle-$ARCH_SUFFIX"
    ORACLEJDK7_JAVA_HOME="/usr/lib/jvm/java-7-oracle-$ARCH_SUFFIX"
else
    ORACLEJDK7_UJA_ALIAS="java-7-oracle"
    ORACLEJDK7_JAVA_HOME="/usr/lib/jvm/java-7-oracle"
fi

# The java::oraclejdk8 recipe from github.com/travis-ci/travis-cookbooks
# takes care of this alias. We take it for granted here. MK.
if [[ -d "/usr/lib/jvm/java-8-oracle-$ARCH_SUFFIX" ]] ; then
    ORACLEJDK8_UJA_ALIAS="java-8-oracle-$ARCH_SUFFIX"
    ORACLEJDK8_JAVA_HOME="/usr/lib/jvm/java-8-oracle-$ARCH_SUFFIX"
else
    ORACLEJDK8_UJA_ALIAS="java-8-oracle"
    ORACLEJDK8_JAVA_HOME="/usr/lib/jvm/java-8-oracle"
fi

UJA="update-java-alternatives"

switch_to_openjdk6 () {
    echo "Switching to OpenJDK6 ($OPENJDK6_UJA_ALIAS), JAVA_HOME will be set to $OPENJDK6_JAVA_HOME"
    sudo $UJA --set "$OPENJDK6_UJA_ALIAS"
    export JAVA_HOME="$OPENJDK6_JAVA_HOME"
}

switch_to_openjdk7 () {
    echo "Switching to OpenJDK7 ($OPENJDK7_UJA_ALIAS), JAVA_HOME will be set to $OPENJDK7_JAVA_HOME"
    sudo $UJA --set "$OPENJDK7_UJA_ALIAS"
    export JAVA_HOME="$OPENJDK7_JAVA_HOME"
}

switch_to_openjdk8 () {
    echo "Switching to OpenJDK8 ($OPENJDK8_UJA_ALIAS), JAVA_HOME will be set to $OPENJDK8_JAVA_HOME"
    sudo $UJA --set "$OPENJDK8_UJA_ALIAS"
    export JAVA_HOME="$OPENJDK8_JAVA_HOME"
}

switch_to_oraclejdk7 () {
    echo "Switching to Oracle JDK7 ($ORACLEJDK7_UJA_ALIAS), JAVA_HOME will be set to $ORACLEJDK7_JAVA_HOME"
    sudo $UJA --set "$ORACLEJDK7_UJA_ALIAS"
    export JAVA_HOME="$ORACLEJDK7_JAVA_HOME"
}

switch_to_oraclejdk8 () {
    echo "Switching to Oracle JDK8 ($ORACLEJDK8_UJA_ALIAS), JAVA_HOME will be set to $ORACLEJDK8_JAVA_HOME"
    sudo $UJA --set "$ORACLEJDK8_UJA_ALIAS"
    export JAVA_HOME="$ORACLEJDK8_JAVA_HOME"
}

print_home_of_openjdk6 () {
    echo "$OPENJDK6_JAVA_HOME"
}

print_home_of_openjdk7 () {
    echo "$OPENJDK7_JAVA_HOME"
}

print_home_of_openjdk8 () {
    echo "$OPENJDK8_JAVA_HOME"
}

print_home_of_oraclejdk7 () {
    echo "$ORACLEJDK7_JAVA_HOME"
}

print_home_of_oraclejdk8 () {
    echo "$ORACLEJDK8_JAVA_HOME"
}

warn_sunjdk6_eol () {
    echo "Sun/Oracle JDK 6 is EOL since November 2012, and is no longer supported. Please consider upgrading..." >&2
}

warn_jdk_not_known () {
    echo "Sorry, but JDK '$1' is not known." >&2
}

warn_gcj_user () {
    echo "We do not support GCJ. I mean, come on. Are you Richard Stallman?" >&2
}

switch_jdk()
{
    case "${1:-default}" in
        *gcj*)
            warn_gcj_user
            false
            ;;
        openjdk6|openjdk1.6|openjdk1.6.0|jdk6|1.6.0|1.6|6.0)
            switch_to_openjdk6
            ;;
        openjdk7|jdk7|1.7.0|1.7|7.0)
            switch_to_openjdk7
            ;;
        openjdk8|jdk8|1.8.0|1.8|8.0)
            switch_to_openjdk8
            ;;
        oraclejdk6|oraclejdk1.6|oraclejdk1.6.0|oracle6|oracle1.6.0|oracle6.0|sunjdk6|sun6)
            warn_sunjdk6_eol
            false
            ;;
        oraclejdk7|oraclejdk1.7|oraclejdk1.7.0|oracle7|oracle1.7.0|oracle7.0|oracle|sunjdk7|sun7|sun)
            switch_to_oraclejdk7
            ;;
        oraclejdk8|oraclejdk1.8|oraclejdk1.8.0|oracle8|oracle1.8.0|oracle8.0)
            switch_to_oraclejdk8
            ;;
        default)
            switch_to_oraclejdk7
            ;;
        *)
            warn_jdk_not_known "$1"
            false
            ;;
    esac
}


print_java_home()
{
    typeset JDK
    JDK="$1"

    case "$JDK" in
        *gcj*)
            warn_gcj_user
            ;;
        openjdk6|openjdk1.6|openjdk1.6.0|jdk6|1.6.0|1.6|6.0)
            print_home_of_openjdk6
            ;;
        openjdk7|jdk7|1.7.0|1.7|7.0)
            print_home_of_openjdk7
            ;;
        openjdk8|jdk8|1.8.0|1.8|8.0)
            print_home_of_openjdk8
            ;;	    
        oraclejdk6|oraclejdk1.6|oraclejdk1.6.0|oracle6|oracle1.6.0|oracle6.0|sunjdk6|sun6)
            warn_sunjdk6_eol
            ;;
        oraclejdk7|oraclejdk1.7|oraclejdk1.7.0|oracle7|oracle1.7.0|oracle7.0|oracle|sunjdk7|sun7|sun)
            print_home_of_oraclejdk7
            ;;
        oraclejdk8|oraclejdk1.8|oraclejdk1.8.0|oracle8|oracle1.8.0|oracle8.0)
            print_home_of_oraclejdk8
            ;;
        default)
            print_home_of_openjdk7
            ;;
        *)
            warn_jdk_not_known "$JDK"
            ;;
    esac
}


jdk_switcher()
{
    typeset COMMAND JDK
    COMMAND="$1"
    JDK="$2"

    case "$COMMAND" in
        use)
            switch_jdk "$JDK"
            ;;
        home)
            print_java_home "$JDK"
            ;;
        *)
            echo "Usage: $0 {use|home} [ JDK version ]" >&2
            false
            ;;
    esac

    return $?
}
