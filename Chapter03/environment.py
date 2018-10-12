'''
Created on Mar 25, 2018

@author: ywz
'''
from threading import Thread


def new_demo(test=True):
    import pygame
    from demo.game import Game
    
    if test is False:
        game = Game(640, 480, None)
    else:
        def _render(game):
            while True:
                game.draw()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_9:
                            game.increase_fps()
                        elif event.key == pygame.K_0:
                            game.decrease_fps()    
        pygame.init()
        DISPLAYSURF = pygame.display.set_mode((640, 480), 0, 32)
        pygame.display.set_caption('Demo')
        game = Game(640, 480, DISPLAYSURF)
        t = Thread(target=lambda: _render(game))
        t.start()
    
    return game


def new_atari_game(rom='breakout'):
    from game import Game
    
    game = Game(rom)
    
    if rom == 'space_invaders':
        game.set_params(frame_skip=3, lost_life_as_terminal=False, take_maximum_of_two_frames=True)
    elif game == 'alien':
        game.set_params(frame_skip=4, crop_offset=20, lost_life_as_terminal=False)
    else:
        game.set_params(frame_skip=4, lost_life_as_terminal=False)
        
    return game

