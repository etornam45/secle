export type Shape = number[];

type FileContent = {
  imports: string[];
  code: string; // code without import statements
}

function readCode(code: string): FileContent {
  const importRegex = /import\s+(?:\w+|{\s*[^}]+\s*})\s+from\s+['"]([^'"]+)['"]\s*;?/g;
  const exportFromRegex = /export\s+(?:\*|{\s*[^}]+\s*})\s+from\s+['"]([^'"]+)['"]\s*;?/g;
  const imports = [];
  let match;

  while ((match = importRegex.exec(code))) {
    imports.push(match[1]);
  }

  while ((match = exportFromRegex.exec(code))) {
    imports.push(match[1]);
  }

  return {
    imports,
    code: code.replace(importRegex, "").replace(exportFromRegex, "").trim()
  };
}

function readFileContent(filePath: string): string {
  try {
    console.log(`Reading file: ${filePath}`);
    const data = Deno.readTextFileSync(filePath);
    return data;
  } catch (error) {
    console.error(`Error reading file: ${filePath}`);
    // throw error;
    return "";
  }
}

function resolvePath(importSpecifier: string, fromPath: string): string {
  if (importSpecifier.startsWith("./") || importSpecifier.startsWith("../")) {
    const base = fromPath.substring(0, fromPath.lastIndexOf("/") + 1);
    return base + importSpecifier.substring(2);
  }
  return importSpecifier;
}

function bundle(entry: string, out: string = "bundled.ts") {
  console.log(`Bundling ${entry} into ${out}...`);
  const visited = new Set<string>();
  const topo: FileContent[] = [];

  function buildTopo(filePath: string) {
    if (visited.has(filePath)) return;
    visited.add(filePath);

    const { imports: nestedImports, code: nestedCode } = readCode(readFileContent(filePath));
    for (const imp of nestedImports) {
      const resolvedPath = resolvePath(imp, filePath);
      buildTopo(resolvedPath);
    }

    topo.push({ imports: nestedImports, code: nestedCode });
  }
  buildTopo(entry);
  console.log(topo.map(file => file.imports).join("\n"));
  console.log(`Bundling ${entry} into ${out}...`);

  Deno.writeTextFileSync(out, topo.map(file => removeComments(file.code)).join("\n"));
}

function removeComments(code: string): string {
  // remove single-line and multi-line comments
  return code.replace(/\/\/.*|\/\*[\s\S]*?\*\//g, "");
}

bundle("index.ts", "dist/bundle.ts");