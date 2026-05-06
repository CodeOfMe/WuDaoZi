"""WuDaoZi GUI - PySide6-based painting studio interface."""

import sys
from pathlib import Path


def run_gui():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is required for GUI mode. Install with: pip install wudaozi[gui]", file=sys.stderr)
        sys.exit(1)

    from PySide6.QtCore import Qt, QThread, Signal, Slot
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    class GenerateWorker(QThread):
        finished = Signal(object)
        error = Signal(str)

        def __init__(self, prompt, config, reference=None):
            super().__init__()
            self.prompt = prompt
            self.config = config
            self.reference = reference

        def run(self):
            try:
                from .core import WuDaoZiEngine

                engine = WuDaoZiEngine(model_path=self.config.get("model_path", ""))
                kwargs = dict(
                    prompt=self.prompt,
                    height=self.config.get("height", 1024),
                    width=self.config.get("width", 1024),
                    num_inference_steps=self.config.get("num_inference_steps", 8),
                    guidance_scale=self.config.get("guidance_scale", 0.0),
                    seed=self.config.get("seed", 42),
                )
                if self.reference:
                    from .core import ReferenceProfile

                    ref = ReferenceProfile(
                        name="gui_ref",
                        style_description=self.reference.get("style", ""),
                        character_description=self.reference.get("character", ""),
                    )
                    img = engine.generate_with_reference(
                        self.prompt, ref, **{k: v for k, v in kwargs.items() if k != "prompt"}
                    )
                else:
                    img = engine.generate(**kwargs)
                self.finished.emit(img)
            except Exception as e:
                self.error.emit(str(e))

    class WuDaoZiWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("WuDaoZi \u5434\u9053\u5b50 - Painting Studio")
            self.setMinimumSize(1200, 800)
            self._worker = None
            self._build_ui()

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)

            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)

            gen_group = QGroupBox("Generation")
            gen_layout = QVBoxLayout(gen_group)
            gen_layout.addWidget(QLabel("Prompt:"))
            self.prompt_edit = QTextEdit()
            self.prompt_edit.setMaximumHeight(120)
            self.prompt_edit.setPlaceholderText("Describe the image you want to paint...")
            gen_layout.addWidget(self.prompt_edit)

            params_layout = QHBoxLayout()
            params_layout.addWidget(QLabel("H:"))
            self.height_spin = QSpinBox()
            self.height_spin.setRange(256, 4096)
            self.height_spin.setSingleStep(256)
            self.height_spin.setValue(1024)
            params_layout.addWidget(self.height_spin)
            params_layout.addWidget(QLabel("W:"))
            self.width_spin = QSpinBox()
            self.width_spin.setRange(256, 4096)
            self.width_spin.setSingleStep(256)
            self.width_spin.setValue(1024)
            params_layout.addWidget(self.width_spin)
            params_layout.addWidget(QLabel("Steps:"))
            self.steps_spin = QSpinBox()
            self.steps_spin.setRange(1, 100)
            self.steps_spin.setValue(8)
            params_layout.addWidget(self.steps_spin)
            params_layout.addWidget(QLabel("Seed:"))
            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 999999)
            self.seed_spin.setValue(42)
            params_layout.addWidget(self.seed_spin)
            gen_layout.addLayout(params_layout)

            self.generate_btn = QPushButton("Generate")
            self.generate_btn.clicked.connect(self._on_generate)
            gen_layout.addWidget(self.generate_btn)
            left_layout.addWidget(gen_group)

            ref_group = QGroupBox("Reference (Style/Character)")
            ref_layout = QVBoxLayout(ref_group)
            ref_layout.addWidget(QLabel("Style:"))
            self.style_edit = QLineEdit()
            self.style_edit.setPlaceholderText("e.g. ink wash painting, traditional Chinese style")
            ref_layout.addWidget(self.style_edit)
            ref_layout.addWidget(QLabel("Character:"))
            self.character_edit = QLineEdit()
            self.character_edit.setPlaceholderText("e.g. a warrior in red armor, flowing cape")
            ref_layout.addWidget(self.character_edit)
            self.ref_file_btn = QPushButton("Load Reference Profile JSON")
            self.ref_file_btn.clicked.connect(self._on_load_ref_file)
            ref_layout.addWidget(self.ref_file_btn)
            left_layout.addWidget(ref_group)

            batch_group = QGroupBox("Batch / Series")
            batch_layout = QVBoxLayout(batch_group)
            self.prompts_list = QListWidget()
            batch_layout.addWidget(QLabel("Prompts (one per line):"))
            batch_layout.addWidget(self.prompts_list)
            batch_btn_layout = QHBoxLayout()
            self.load_prompts_btn = QPushButton("Load Prompts File")
            self.load_prompts_btn.clicked.connect(self._on_load_prompts)
            batch_btn_layout.addWidget(self.load_prompts_btn)
            self.add_prompt_btn = QPushButton("Add Current Prompt")
            self.add_prompt_btn.clicked.connect(self._on_add_prompt)
            batch_btn_layout.addWidget(self.add_prompt_btn)
            self.clear_prompts_btn = QPushButton("Clear")
            self.clear_prompts_btn.clicked.connect(self.prompts_list.clear)
            batch_btn_layout.addWidget(self.clear_prompts_btn)
            batch_layout.addLayout(batch_btn_layout)
            self.series_name_edit = QLineEdit()
            self.series_name_edit.setPlaceholderText("Series name")
            batch_layout.addWidget(QLabel("Series Name:"))
            batch_layout.addWidget(self.series_name_edit)
            self.batch_btn = QPushButton("Batch Generate")
            self.batch_btn.clicked.connect(self._on_batch)
            batch_layout.addWidget(self.batch_btn)
            left_layout.addWidget(batch_group)

            left_layout.addStretch()
            left_panel.setMinimumWidth(400)
            left_panel.setMaximumWidth(500)

            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            right_layout.addWidget(QLabel("Preview:"))
            self.preview_label = QLabel("No image yet")
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumSize(512, 512)
            self.preview_label.setStyleSheet("border: 1px solid #ccc; background: #1a1a1a;")
            right_layout.addWidget(self.preview_label)
            right_layout.addStretch()

            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
            main_layout.addWidget(splitter)

            self.statusBar().showMessage("Ready - WuDaoZi Painting Studio")

        def _get_config(self):
            return {
                "height": self.height_spin.value(),
                "width": self.width_spin.value(),
                "num_inference_steps": self.steps_spin.value(),
                "guidance_scale": 0.0,
                "seed": self.seed_spin.value(),
                "model_path": "",
            }

        def _get_reference(self):
            style = self.style_edit.text().strip()
            character = self.character_edit.text().strip()
            if style or character:
                return {"style": style, "character": character}
            return None

        def _on_generate(self):
            prompt = self.prompt_edit.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(self, "Warning", "Please enter a prompt.")
                return
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("Generating...")
            self.statusBar().showMessage("Generating image...")
            self._worker = GenerateWorker(prompt, self._get_config(), self._get_reference())
            self._worker.finished.connect(self._on_image_ready)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        @Slot(object)
        def _on_image_ready(self, img):
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate")
            self.statusBar().showMessage("Image generated successfully")
            from PIL.ImageQt import ImageQt

            qimg = ImageQt(img)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        @Slot(str)
        def _on_error(self, msg):
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate")
            self.statusBar().showMessage(f"Error: {msg}")
            QMessageBox.critical(self, "Error", msg)

        def _on_add_prompt(self):
            prompt = self.prompt_edit.toPlainText().strip()
            if prompt:
                self.prompts_list.addItem(prompt)

        def _on_load_prompts(self):
            path, _ = QFileDialog.getOpenFileName(self, "Load Prompts File", "", "Text Files (*.txt);;All Files (*)")
            if path:
                from .core import read_prompts

                try:
                    prompts = read_prompts(path)
                    self.prompts_list.clear()
                    for p in prompts:
                        self.prompts_list.addItem(p)
                    self.statusBar().showMessage(f"Loaded {len(prompts)} prompts from {path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))

        def _on_load_ref_file(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Reference Profile", "", "JSON Files (*.json);;All Files (*)"
            )
            if path:
                try:
                    from .core import load_reference_profile

                    ref = load_reference_profile(name="gui", from_file=path)
                    self.style_edit.setText(ref.style_description)
                    self.character_edit.setText(ref.character_description)
                    self.statusBar().showMessage(f"Loaded reference: {ref.name}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))

        def _on_batch(self):
            if self.prompts_list.count() == 0:
                QMessageBox.warning(self, "Warning", "Please add prompts first.")
                return
            prompts = [self.prompts_list.item(i).text() for i in range(self.prompts_list.count())]
            self.series_name_edit.text().strip() or "untitled"
            self.statusBar().showMessage(f"Batch generating {len(prompts)} images...")
            self._batch_prompts = prompts
            self._batch_index = 0
            self._batch_results = []
            self._batch_next()

        def _batch_next(self):
            if self._batch_index >= len(self._batch_prompts):
                self.statusBar().showMessage(f"Batch complete: {len(self._batch_results)} images generated")
                return
            prompt = self._batch_prompts[self._batch_index]
            self.statusBar().showMessage(f"Generating {self._batch_index + 1}/{len(self._batch_prompts)}...")
            self._worker = GenerateWorker(prompt, self._get_config(), self._get_reference())
            self._worker.finished.connect(self._on_batch_image)
            self._worker.error.connect(self._on_error)
            self._worker.start()

        @Slot(object)
        def _on_batch_image(self, img):
            series_name = self.series_name_edit.text().strip() or "untitled"
            from .core import slugify

            output_dir = Path("outputs") / slugify(series_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self._batch_index + 1:03d}.png"
            img.save(str(output_dir / filename))
            self._batch_results.append(str(output_dir / filename))
            self._batch_index += 1
            from PIL.ImageQt import ImageQt

            qimg = ImageQt(img)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self._batch_next()

    app = QApplication.instance() or QApplication(sys.argv)
    window = WuDaoZiWindow()
    window.show()
    sys.exit(app.exec())
