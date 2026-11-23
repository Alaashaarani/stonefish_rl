"""Open the RL workspace or the control window for an active multi-training job."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Load this task YAML into the workspace.')
    parser.add_argument('--tab', choices=('tasks', 'test', 'evaluate', 'plot'), default='tasks')
    parser.add_argument('--multi-control', metavar='DIRECTORY', help='Monitor/control an existing multi-training job.')
    args = parser.parse_args(argv)
    if args.multi_control:
        from library.multi_train_control_gui import MultiTrainControlGui
        app = MultiTrainControlGui(args.multi_control)
    else:
        from library.gui_application import StonefishRLGui
        app = StonefishRLGui()
        if args.config:
            app.tasks.config_path.set(args.config)
            app.testing.config_path.set(args.config)
            app.testing.load_yaml()
            app.evaluation.config_path.set(args.config)
            app.evaluation.load_config(load_models=True)
        panels = {'tasks': app.tasks, 'test': app.testing, 'evaluate': app.evaluation, 'plot': app.plot_tab}
        app.tabs.select(panels[args.tab])
    app.mainloop()


if __name__ == '__main__':
    main()
