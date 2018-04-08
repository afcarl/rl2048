from matplotlib import pyplot as plt
import sys
import pickle


ax_loss = plt.subplot(221)
ax_training_maxblock = plt.subplot(222)
ax_validation_maxblock = plt.subplot(223, sharex=ax_loss)
ax_validation_validmoves = plt.subplot(224, sharex=ax_loss)

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        stats = pickle.load(f)

        ax_loss.plot(stats['training_step'], stats['loss'])
        ax_loss.set_xlabel('Steps')
        ax_loss.set_ylabel('Loss')

        ax_training_maxblock.plot(stats['episode'],
                                  stats['training_max_block'])
        ax_training_maxblock.set_yscale('log', basey=2)
        ax_training_maxblock.set_xlabel('Training Max. Block')
        ax_training_maxblock.set_ylabel('Episode')

        ax_validation_maxblock.plot(stats['validation_step'],
                                    stats['validation_max_block'])
        ax_validation_maxblock.set_xlabel('Steps')
        ax_validation_maxblock.set_ylabel('Validation Max. block')

        ax_validation_validmoves.plot(stats['validation_step'],
                                      stats['validation_valid_moves'])
        ax_validation_validmoves.set_ylabel('Steps')
        ax_validation_validmoves.set_ylabel('Validation Valid moves')

        plt.show()
