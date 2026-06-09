import sys
path = r'C:\supreme-chainsaw\analysis\gradient_flow_analyzer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the __init__ method
old_init = (
    '    def __init__(self, verbose=0):\n'
    '        super().__init__(verbose)\n'
    '        self.epoch = 0'
)
new_init = (
    '    def __init__(self, verbose=0, pretrain_loss_reduction=0.0):\n'
    '        super().__init__(verbose)\n'
    '        self.epoch = 0\n'
    '        self.pretrain_loss_reduction = pretrain_loss_reduction'
)
if old_init in content:
    content = content.replace(old_init, new_init, 1)
    print('OK: __init__ updated with pretrain_loss_reduction param')
else:
    print('WARN: __init__ pattern not found!')

# Add _on_training_start method before _on_step
old_on_step = (
    '    def _on_step(self) -> bool:'
)
new_on_step = (
    '    def _on_training_start(self) -> None:\n'
    '        if self.pretrain_loss_reduction != 0.0:\n'
    '            writer.add_scalar("pretrain/loss_reduction_pct", self.pretrain_loss_reduction, 0)\n'
    '            writer.flush()\n'
    '\n'
    '    def _on_step(self) -> bool:'
)
if old_on_step in content:
    content = content.replace(old_on_step, new_on_step, 1)
    print('OK: _on_training_start method added')
else:
    print('WARN: _on_step pattern not found!')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: gradient_flow_analyzer.py updated')
