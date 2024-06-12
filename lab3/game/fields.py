import pygame


class Field:
    """
    Represents a single field on the game board.

    Attributes:
    -----------
    window : pygame.Surface
        The surface on which the field will be drawn.
    row : int
        The row num of the field on the game board.
    col : int
        The column num of the field on the game board.
    draw_params : dict
        A dictionary containing params for drawing the field.
    is_drawed : bool
        A flag indicating whether the field has been drawn.
    """
    def __init__(self, window, row, col, draw_params):
        self.window, self.row, self.col, self.draw_params, self.is_drawed = window, row, col, draw_params, False

    def draw(self):
        self.is_drawed = True


class BlockedField(Field):
    """
    A class representing a blocked field in the isolation game.

    Inherits from the Field class and overrides the draw method to draw a blocked field.
    """
    def draw(self):
        y, x = self.row * self.draw_params.field_size, self.col * self.draw_params.field_size
        pygame.draw.rect(self.window, self.draw_params.block_field_color, (x, y, self.draw_params.field_size, self.draw_params.field_size))
        self.is_drawed = True


class AvailableField(Field):
    """
    Represents a field that is available for a player to move to.
    Inherits from the Field class.
    """
    def draw(self):
        y, x = self.row * self.draw_params.field_size, self.col * self.draw_params.field_size
        x_with_padding, y_with_padding = x + self.draw_params.padding, y + self.draw_params.padding
        available_field_size = (self.draw_params.field_size - self.draw_params.padding)
        pygame.draw.rect(self.window, self.draw_params.block_field_color, (x, y, self.draw_params.field_size, self.draw_params.field_size))
        pygame.draw.rect(self.window, self.draw_params.available_field_color, (x_with_padding, y_with_padding, available_field_size, available_field_size))
        self.is_drawed = True


class P1Field(Field):
    """
    A class representing a field on the game board for the first player.

    Inherits from the `Field` class and overrides the `draw` method to draw the field
    with the first player's color and a circle representing the first player's pos.
    """
    def draw(self):
        y, x = self.row * self.draw_params.field_size, self.col * self.draw_params.field_size
        x_with_padding, y_with_padding = x + self.draw_params.padding, y + self.draw_params.padding
        available_field_size = (self.draw_params.field_size - self.draw_params.padding)
        x_player, y_player = x_with_padding + (available_field_size / 2), y_with_padding + (available_field_size / 2)
        pygame.draw.rect(self.window, self.draw_params.block_field_color, (x, y, self.draw_params.field_size, self.draw_params.field_size))
        pygame.draw.rect(self.window, self.draw_params.available_field_color, (x_with_padding, y_with_padding, available_field_size, available_field_size))
        pygame.draw.circle(self.window, self.draw_params.p1_color, (x_player, y_player), self.draw_params.player_size)
        self.is_drawed = True


class P2Field(Field):
    """
    A class representing a field on the game board for the second player.

    Inherits from the Field class and implements the draw method to draw the field on the game board.
    """
    def draw(self):
        y, x = self.row * self.draw_params.field_size, self.col * self.draw_params.field_size
        x_with_padding, y_with_padding = x + self.draw_params.padding, y + self.draw_params.padding
        available_field_size = (self.draw_params.field_size - self.draw_params.padding)
        x_player, y_player = x_with_padding + (available_field_size / 2), y_with_padding + (available_field_size / 2)
        pygame.draw.rect(self.window, self.draw_params.block_field_color, (x, y, self.draw_params.field_size, self.draw_params.field_size))
        pygame.draw.rect(self.window, self.draw_params.available_field_color, (x_with_padding, y_with_padding, available_field_size, available_field_size))
        pygame.draw.circle(self.window, self.draw_params.p2_color, (x_player, y_player), self.draw_params.player_size)
        self.is_drawed = True
