class Notifier:
    def __init__(self):
        pass

    def notify(self, cat_name, duration):
        print(f"{cat_name} finished using the sandbox. Duration: {duration} seconds.")