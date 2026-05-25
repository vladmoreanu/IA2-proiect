from lighter import Config

cfg = Config({
    "a": 1,
    "b": 2,
    "nested": {"x": 10}
})

cfg["a", "b"] = [0, 1]
cfg["nested.x"] = 99

print(cfg)
print({**cfg.nested})
print(cfg.nested.x)