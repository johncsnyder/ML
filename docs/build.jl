
using Docile, Lexicon, ML

# config = Config()
config = Config(md_subheader=:skip, mathjax = true)

index = save("ML.md", ML, config)
save("index.md", Index([index]), config)

# run(`../mkdocs build`)
# mkdocs gh-deploy --clean

