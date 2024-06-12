import pygame
from q_uber import read_lab_from_file, Musk_Taxi
from PIL import Image

BLUE = (50, 100, 250)
GREEN = (45, 200, 35)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
SCALE_FACTOR = 0.94
WIDTH = HEIGHT = 840
FPS = 5
VISUALIZATION_BIAS = 0.03
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Uber")


class Game:
    """
    Represents a game instance.

    Args:
        win (pygame.Surface): The game window surface.
        taxi (Taxi): The taxi object.
        car_image_path (str): The file path of the car image.
        student_image_path (str): The file path of the student image.
    """
    def __init__(self, win, taxi, car_image_path, student_image_path, steps):
        self.win = win
        self.taxi = taxi
        self.size = HEIGHT/self.taxi.rows
        self.steps = steps
        self.iter = 0
        self.car_image = self.image_resize(car_image_path, SCALE_FACTOR)
        self.student_image = self.image_resize(student_image_path, SCALE_FACTOR)
        self.trail = []

    def image_resize(self, path, factor):
        """
        Resizes an image and returns the resized image.

        Args:
            path (str): The file path of the image.
            factor (float): The scaling factor.

        Returns:
            pygame.Surface: The resized image surface.
        """
        image = Image.open(path)
        if image.size[0] != int(self.size) and image.size[1] != int(self.size):
            new_image = image.resize((int(self.size*factor), int(self.size*factor)))
            new_image.save(path)
            return pygame.image.load(path)
        return pygame.image.load(path)

    def update(self):
        self.win.fill(BLACK)
        self.trail.append((self.taxi.y, self.taxi.x))
        for row in range(0, self.taxi.rows):
            for col in range(0, self.taxi.rows):
                pygame.draw.rect(self.win, BLUE, ((col + VISUALIZATION_BIAS) * self.size, (row + VISUALIZATION_BIAS) * self.size, self.size*SCALE_FACTOR, self.size*SCALE_FACTOR))
        for x in range(self.taxi.rows):
            for y in range(self.taxi.rows):
                if self.taxi.lab[x][y] == 1:
                    self.win.blit(self.student_image, ((y + VISUALIZATION_BIAS) * self.size, (x + VISUALIZATION_BIAS) * self.size))
        pygame.draw.circle(self.win, BLACK, ((self.taxi.y_done + 0.5) * self.size, (self.taxi.x_done + 0.5) * self.size), self.size/2.8)
        pygame.draw.circle(self.win, RED, ((self.taxi.y_done + 0.5) * self.size, (self.taxi.x_done + 0.5) * self.size), self.size/3)
        for pos in self.trail:
            pygame.draw.rect(self.win, GREEN, ((pos[0] + VISUALIZATION_BIAS) * self.size, (pos[1] + VISUALIZATION_BIAS) * self.size, self.size*SCALE_FACTOR, self.size*SCALE_FACTOR))
        self.win.blit(self.car_image, ((self.taxi.y + VISUALIZATION_BIAS) * self.size, (self.taxi.x + VISUALIZATION_BIAS) * self.size))

        self.taxi.make_action(self.steps[self.iter])
        if self.iter < len(self.steps) - 1:
            self.iter += 1


def main():
    run = True
    clock = pygame.time.Clock()

    lab = read_lab_from_file('test2.txt')
    steps = [2, 3, 2, 2, 2, 3, 3, 3, 0, 3, 0, 0, 0, 3, 3]
    taxi = Musk_Taxi(lab, (7, 0))
    game = Game(WIN, taxi, "car.jpg", "stażysta.jpg", steps)
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        game.update()
        pygame.display.update()
    pygame.quit()


if __name__ == "__main__":
    main()
