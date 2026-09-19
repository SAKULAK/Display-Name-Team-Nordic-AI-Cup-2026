"""Optional evaluation-only viewer; no controller logic or physics changes."""


class Viewer:
    def __init__(self, env, seed):
        import pygame
        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        info = pygame.display.Info()
        scale = min(1., info.current_w * .9 / env.width, info.current_h * .85 / env.height)
        self.screen = pygame.display.set_mode((int(env.width * scale), int(env.height * scale)))
        pygame.display.set_caption(f"Experimental shared PPO | seed {seed}")
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

    def draw(self, env):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return False
        caches = [(a, a._vision_poly) for a in list(env.agents) + list(env.predators)]
        try:
            env.draw(self.screen)
        finally:
            for a, cache in caches:
                a._vision_poly = cache
        text = self.font.render(f"Shared PPO | t={env.time:.1f}s | score={env.score:.3f} | population={len(env.agents)}", True, (255, 255, 255), (10, 10, 15))
        self.screen.blit(text, (10, 10))
        pg.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        self.pg.display.quit()
        self.pg.font.quit()
