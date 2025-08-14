# The purpose of this MCP server

I am a software developer working with IFS Cloud daily, making customizations and tailoring the application to our business' need.
I want to be able to move faster, shipping more features with less errors.

## Analysis of the core application code

A significant time is spent on analyzing the core codebase from IFS. It contains the entire application code, both frontend (in the form of marble scripts (.client, .projection, .fragment) and supporting PL/SQL scripts (.plsvc)) and backend (in the form of entity definitions (.entity), index and sequence definitions (.storage), oracle views (.views), and oracle plsql functions/procedures (.plsql)).

These files have different syntaxes:

- .entity uses XML
- .storage uses a DSL language that mimics a tiny subset of Oracle DDL
- .views uses a DSL language that closely matches Oracle SQL, but with syntactic sugar that makes creating column comments easier (these contain metadata about the columns, such as labels, datatype, references, etc.) and overall just making it a bit more concise. We also have the possibility to @Override or @Overtake functions and procedures from the layer below (in our case, it's the IFS Cloud Core layer)
- .plsql is almost identical to Oracle PL/SQL, but with some extra code generation attributes and not having to write CREATE PACKAGE, we only write the actual functions and procedures. We also have the possibility to @Override or @Overtake functions and procedures from the layer below (in our case, it's the IFS Cloud Core layer)
- .plsvc is the same as .plsql, but it's used to support its accompanying .projection file. You can @Override and @Overtake here as well. Although you don't always need a .plsql file. If you just keep it simple and don't add any commands to the projection, then you can create entire pages without having to write PL/SQL.
- .projection defines the data access layer, which is an OData v4 API. This API is generated from the projection definition. This modelling language is called "Marble".
- .client defines the front-end layout/logic (although a bit limited, I try to stay away from it to keep things simple and not have business logic in the frontend). The modelling language for the client file is also "Marble".
- .fragment is a reusable piece of "Marble" code, which can contain both projection and client code. An important thing to note, is that when a fragment file is "included" in a projection or client file, it searches it globally across all modules. This means that every fragment file must have a globally unique name!

Analyzing these are tedious and time consuming. Having an analyzer would speed things up, and possibly eliminate the need for me to manually going through tons of code.

## Indexing

This analyzer should be used together with an indexer and exposed as an MCP server, which would allow AI coding agents to use that insight to better implement the business requirements as code in our customization project.

## Sample data

Creating sample data by hand would be tedious, so what I've done is create a powershell script that extracts the aforementioned file types that we are interested in, and places them in the "./\_work/" directory, while preserving their original directory structure.

## Building the MCP server

When you are building the MCP server code, follow these rules:

- You should use the code from the "./\_work/" directory to analyze each file type and their intricacies.
- You should always strive to create generic solutions, even if that means not passing every single test that you create. Simple, generic solutions are always better, especially when we're going to maintain and extend the code in the future.
- If you create temporary files for iteration and testing, and they're not part of the final solution, clean up after yourself. I don't want unused files in the project.
- Functions should have documentation, but don't overdo it
- Don't be afraid to try things out; go all out!

## Using the MCP server

After the MCP server is built, we should be able to tell it where the "IFS Cloud Core Codes" are on the file system. That way, we're able to stay up-to-date with IFS whenever they release new versions of their core codes, without having to re-create the entire MCP server.
