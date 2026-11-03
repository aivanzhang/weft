// Separate argument parsing lets fixtures verify explicit mode without starting a browser.
export function retryOptions(args) {
  if (args.length < 2 || args.length > 3 || !/^[a-f0-9]{24}$/.test(args[1]) ||
      args.length === 3 && args[2] !== "--preserve-current") {
    throw new Error("Require new output folder, terminal parent ID and optional --preserve-current");
  }
  const preserveCurrent = args.length === 3;
  return { folder: args[0], parent: args[1], preserveCurrent,
    button: preserveCurrent ? "#preserve-current-job" : "#retry-job",
    body: preserveCurrent ? { appearance_policy: "preserve_current" } : {} };
}
