// CodeMirror Marble language mode for IFS Cloud
// Based on the TextMate grammar converted to CodeMirror syntax

(function (mod) {
  if (typeof exports == "object" && typeof module == "object")
    // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd)
    // AMD
    define(["../../lib/codemirror"], mod);
  // Plain browser env
  else mod(CodeMirror);
})(function (CodeMirror) {
  "use strict";

  CodeMirror.defineMode("marble", function (config, parserConfig) {
    // IFS Cloud Keywords
    const keywords = {
      // Control flow
      if: true,
      else: true,
      elif: true,
      endif: true,
      while: true,
      for: true,
      foreach: true,
      do: true,
      loop: true,
      break: true,
      continue: true,
      return: true,
      try: true,
      catch: true,
      finally: true,
      throw: true,

      // IFS Declarations
      entity: true,
      view: true,
      projection: true,
      fragment: true,
      client: true,
      api: true,
      service: true,
      procedure: true,
      function: true,
      method: true,
      class: true,
      interface: true,
      enum: true,
      type: true,
      namespace: true,
      module: true,

      // IFS UI Components
      page: true,
      list: true,
      group: true,
      iconset: true,
      tree: true,
      navigator: true,
      selector: true,
      command: true,
      dialog: true,
      assistant: true,
      step: true,
      lov: true,

      // Import/Include
      import: true,
      export: true,
      include: true,
      using: true,
      with: true,
      from: true,

      // Query keywords
      select: true,
      where: true,
      orderby: true,
      groupby: true,
      query: true,
      entityset: true,
      structure: true,

      // Modifiers
      public: true,
      private: true,
      protected: true,
      internal: true,
      static: true,
      readonly: true,
      const: true,
      virtual: true,
      override: true,
      abstract: true,
      sealed: true,
      partial: true,
      async: true,
      await: true,

      // IFS specific modifiers
      editable: true,
      visible: true,
      enabled: true,
      required: true,
      validate: true,
      execute: true,
      call: true,
      refresh: true,

      // Operators
      and: true,
      or: true,
      not: true,
      is: true,
      as: true,
      new: true,
      this: true,
      base: true,
      super: true,
      self: true,

      // Constants
      true: true,
      false: true,
      null: true,
      undefined: true,
    };

    // IFS Data Types
    const types = {
      string: true,
      int: true,
      integer: true,
      long: true,
      short: true,
      byte: true,
      bool: true,
      boolean: true,
      float: true,
      double: true,
      decimal: true,
      char: true,
      void: true,
      object: true,
      any: true,
      var: true,
      Date: true,
      DateTime: true,
      Time: true,
      Guid: true,
      Text: true,
      Number: true,
      Timestamp: true,
      Enumeration: true,
      Binary: true,
      Stream: true,
      Clob: true,
      Blob: true,
      Objid: true,
      Objversion: true,
      State: true,
      List: true,
      Array: true,
      Dictionary: true,
      Collection: true,
    };

    function tokenBase(stream, state) {
      var ch = stream.next();

      // Comments
      if (ch == "/" && stream.eat("/")) {
        stream.skipToEnd();
        return "comment";
      }
      if (ch == "/" && stream.eat("*")) {
        state.tokenize = tokenComment;
        return tokenComment(stream, state);
      }
      if (ch == "#") {
        stream.skipToEnd();
        return "comment";
      }

      // Strings
      if (ch == '"') {
        state.tokenize = tokenString(ch);
        return state.tokenize(stream, state);
      }
      if (ch == "'") {
        state.tokenize = tokenString(ch);
        return state.tokenize(stream, state);
      }

      // Template strings
      if (ch == "$" && stream.eat('"')) {
        state.tokenize = tokenTemplateString;
        return tokenTemplateString(stream, state);
      }

      // Numbers
      if (/\d/.test(ch)) {
        stream.eatWhile(/[\d.]/);
        if (stream.eat(/[eE]/)) {
          stream.eat(/[+-]/);
          stream.eatWhile(/\d/);
        }
        return "number";
      }

      // Hex numbers
      if (ch == "0" && stream.eat(/[xX]/)) {
        stream.eatWhile(/[0-9a-fA-F]/);
        return "number";
      }

      // Annotations
      if (ch == "@") {
        stream.eatWhile(/\w/);
        return "meta";
      }

      // Special variables
      if (ch == "$") {
        stream.eatWhile(/\w/);
        return "variable-2";
      }
      if (ch == "&") {
        stream.eatWhile(/\w/);
        return "variable-2";
      }
      if (ch == "@" && stream.eat("@")) {
        stream.eatWhile(/\w/);
        return "variable-2";
      }

      // Operators
      if (/[+\-*/%&|^~<>!]/.test(ch)) {
        stream.eatWhile(/[+\-*/%&|^~<>!=]/);
        return "operator";
      }

      // Punctuation
      if (/[{}[\]();,.]/.test(ch)) {
        return "bracket";
      }

      // Identifiers and keywords
      if (/[a-zA-Z_]/.test(ch)) {
        stream.eatWhile(/[a-zA-Z0-9_]/);
        var word = stream.current();

        // Check for entity names (PascalCase)
        if (/^[A-Z][a-zA-Z0-9]*$/.test(word)) {
          return "def";
        }

        // Check for constants (ALL_CAPS)
        if (/^[A-Z][A-Z0-9_]*$/.test(word)) {
          return "atom";
        }

        // Check keywords and types
        if (keywords.hasOwnProperty(word)) {
          return "keyword";
        }
        if (types.hasOwnProperty(word)) {
          return "type";
        }

        return "variable";
      }

      return null;
    }

    function tokenString(quote) {
      return function (stream, state) {
        var escaped = false,
          next,
          end = false;
        while ((next = stream.next()) != null) {
          if (next == quote && !escaped) {
            end = true;
            break;
          }
          if (next == "{" && !escaped && quote == '"') {
            // Template string interpolation
            state.tokenize = tokenBase;
            return "string";
          }
          escaped = !escaped && next == "\\";
        }
        if (end || !escaped) state.tokenize = null;
        return "string";
      };
    }

    function tokenTemplateString(stream, state) {
      var ch;
      while ((ch = stream.next()) != null) {
        if (ch == '"') {
          state.tokenize = null;
          return "string";
        }
        if (ch == "{") {
          state.tokenize = tokenBase;
          return "string";
        }
        if (ch == "\\" && stream.next() == null) break;
      }
      return "string";
    }

    function tokenComment(stream, state) {
      var maybeEnd = false,
        ch;
      while ((ch = stream.next()) != null) {
        if (ch == "/" && maybeEnd) {
          state.tokenize = null;
          break;
        }
        maybeEnd = ch == "*";
      }
      return "comment";
    }

    return {
      startState: function () {
        return { tokenize: null };
      },

      token: function (stream, state) {
        if (stream.eatSpace()) return null;
        var style = (state.tokenize || tokenBase)(stream, state);
        return style;
      },

      indent: function (state, textAfter) {
        var firstChar = textAfter && textAfter.charAt(0);
        if (firstChar == "}" || firstChar == "]" || firstChar == ")") {
          return config.indentUnit * (state.indentLevel - 1);
        }
        return config.indentUnit * state.indentLevel;
      },

      lineComment: "//",
      blockCommentStart: "/*",
      blockCommentEnd: "*/",
      fold: "brace",
    };
  });

  CodeMirror.defineMIME("text/x-marble", "marble");
  CodeMirror.defineMIME("text/x-ifs-client", "marble");
  CodeMirror.defineMIME("text/x-ifs-fragment", "marble");
  CodeMirror.defineMIME("text/x-ifs-projection", "marble");
});
