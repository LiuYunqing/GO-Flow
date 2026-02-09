import torch
from .hd import EquivariantHeatDissipation
from .hier_enc import HierarchicalMessagePassing


def get_model(config, args):
    if args.type == 'go_flow':
        from .go_flow import GOFlow
        return GOFlow(config, args)
    else:
        encoder = HierarchicalMessagePassing(args)
        if args.type == 'hd':
            return EquivariantHeatDissipation(config, args, encoder)
        elif args.type == 'enhanced_hd':
            from .enhanced_hd import EnhancedEquivariantHeatDissipation
            return EnhancedEquivariantHeatDissipation(config, args, encoder)
        else:
            raise NotImplementedError('Unknown model: %s' % args.type)
