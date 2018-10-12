'''
Created on 2 Jun 2017

@author: ywz
'''
import time
from threading import Thread
from parameter import Parameter


def new_demo(test=False):
    
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
    
    parameter = Parameter(lr=1e-3)
    parameter.gamma = 0.9
    parameter.iteration_num = 300000
    parameter.num_history_frames = 1
    parameter.network_type = 'mlp'
    
    parameter.update_method = 'rmsprop'
    parameter.rho = 0.95
    parameter.async_update_interval = 5
    parameter.input_scale = 1.0
    
    return game, parameter


def new_atari_game(rom='breakout'):
    
    from game import Game
    game = Game(rom)
    
    if rom == 'space_invaders':
        game.set_params(frame_skip=3, lost_life_as_terminal=False, take_maximum_of_two_frames=True)
    elif game == 'alien':
        game.set_params(frame_skip=4, crop_offset=20, lost_life_as_terminal=False)
    else:
        game.set_params(frame_skip=4, lost_life_as_terminal=False)
    
    parameter = Parameter(lr=7e-4)
    parameter.gamma = 0.99
    parameter.num_history_frames = 4
    
    parameter.async_update_interval = 20
    parameter.max_iter_num = 16 * 10 ** 7
    parameter.update_method = 'rmsprop'
    parameter.rho = 0.99
    parameter.rmsprop_epsilon = 1e-1 # 1e-3 if rom == 'breakout' else 1e-1
    
    time.sleep(1)
    return game, parameter


def new_minecraft(rom='MinecraftBasic-v0'):
    
    from minecraft.game import Game
    game = Game(rom)
    
    parameter = Parameter(lr=7e-4)
    parameter.gamma = 0.99
    parameter.num_history_frames = 4
    
    parameter.async_update_interval = 20
    parameter.max_iter_num = 16 * 10 ** 7
    parameter.update_method = 'rmsprop'
    parameter.rho = 0.99
    parameter.rmsprop_epsilon = 1e-3
    
    time.sleep(1)
    return game, parameter


def new_environment(name='demo', test=False):
    
    if name == 'demo':
        return new_demo(test=test)
    elif name.find('Minecraft') != -1:
        return new_minecraft(rom=name)
    else:
        return new_atari_game(rom=name)


        