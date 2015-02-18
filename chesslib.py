#!/usr/bin/env python
'''
Not a chess engine. Just a simple mechanism to view and replay PGN and FEN files.
'''
import re
import sys
import operator
import pprint
import json
import string
import textwrap

VERSION = (0, 9)

#-------------------------------------------------------------------------------
def get_version():
    return '.'.join(map(str, VERSION))


TRACE = False
EMPTY = '  '
WHITE = 'w'
BLACK = 'b'


# Cached look-up values for computing piece movement
DIAGONAL_VECTORS   = ((1,1), ( 1,-1), (-1, 1), (-1,-1))
ORTHOGONAL_VECTORS = ((1,0), (-1, 0), ( 0, 1), ( 0,-1))

CHECK_TOKEN         = '+'
CHECKMATE_TOKEN     = '#'
ALT_CHECKMATE_TOKEN = '++'
CHECK_TOKENS        = (ALT_CHECKMATE_TOKEN, CHECK_TOKEN, CHECKMATE_TOKEN)

CASTLE_WK  = 0x01
CASTLE_WQ  = 0x02
CASTLE_W   = 0x03
CASTLE_BK  = 0x04
CASTLE_BQ  = 0x08
CASTLE_B   = 0x0C

CASTLE_MAP = dict(K=CASTLE_WK, Q=CASTLE_WQ, k=CASTLE_BK, q=CASTLE_BQ)
CASTLE_FORMAT =((CASTLE_WK, 'K'), (CASTLE_WQ, 'Q'), (CASTLE_BK, 'k'), (CASTLE_BQ, 'q'))


#===============================================================================
class Stream(object):
    
    #---------------------------------------------------------------------------
    def __init__(self, text):
        self.text = text
        self.count = len(text)
        self.index = 0
        
    #---------------------------------------------------------------------------
    def read(self):
        index = self.index + 1
        text = self.text[self.index:index]
        self.index = self.count if index > self.count else index
        return text
    
    #---------------------------------------------------------------------------
    def readline(self):
        substr = self.text[self.index:]
        offset = substr.find('\n')
        if offset > -1:
            self.index += (offset + 1)
            return substr[:offset]
            
        self.index = self.count
        return substr
        
    #---------------------------------------------------------------------------
    def putback(self):
        if self.index > 0:
            self.index -= 1
        else:
            raise ValueError

    #---------------------------------------------------------------------------
    def read_until(self, char):
        chars = ''
        while True:
            c = self.read()
            if not c or c == char:
                return chars

            chars += c

    #---------------------------------------------------------------------------
    def read_tokens(self, chars, validator):
        while True:
            c = self.read()
            if c and validator(c):
                chars += c
            else:
                if c:
                    self.index -= 1
                
                return chars


#===============================================================================
class Ply(object):

    #---------------------------------------------------------------------------
    def __init__(self, symbol, is_white, move_count):
        self.annotations = []
        self.move = symbol
        self.is_white = is_white
        self.move_count = move_count
        self.nag = None
    
    #---------------------------------------------------------------------------
    def add_annotation(self, annotation):
        self.annotations.append(' '.join(annotation.split()))
        
    #---------------------------------------------------------------------------
    def __unicode__(self):
        return '%d%s%s' % (
            self.move_count,
            '. ' if self.is_white else '... ',
            self.move
        )
    __repr__ = __unicode__


#===============================================================================
class Pgn(Ply):

    class Token:
        VALID          = string.ascii_letters + string.digits + '+-/*=#_:?!'
        PERIOD         = '.'
        COMMENT        = ';'
        STRING         = '"'
        RAV_START      = '('
        RAV_END        = ')'
        ANNOTATE_START = '{'
        ANNOTATE_END   = '}'
        TAG_START      = '['
        TAG_END        = ']'
        NAG            = '$'
    
    DEFAULT_FEN    = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    RESULT_STRINGS = ['1/2-1/2', '1-0', '0-1', '*']
    DEFAULT_TAGS   = dict(
        Event='?', 
        Site='?', 
        Date='????.??.??', 
        Round='?', 
        White='?', 
        Black='?', 
        Result ='*'
    )
    
    FEN_RE = re.compile(
        r'''^\s*
            [rnbqkpRNBQKP\d]+(/[rnbqkpRNBQKP\d]+){7}\s  # Piece placement data
            [wb]\s                                      # Active color
            (-|[KQkq]{1,4})\s                           # Castling availability
            (-|[a-h][1-8])\s                            # En passant target square
            \d+\s                                       # Halfmove clock
            \d+\s*                                      # Fullmove number
        $''',
        re.VERBOSE
    )
    
    
    #---------------------------------------------------------------------------
    def __init__(self):
        super(Pgn, self).__init__(None, True, 0)
        self.tags = dict(self.DEFAULT_TAGS, FEN=self.DEFAULT_FEN)
        self.moves = []
        self.result = '*'
        self.next_ply = self

    #---------------------------------------------------------------------------
    def __iter__(self):
        return iter(self.moves)
        
    #---------------------------------------------------------------------------
    def __setitem__(self, attr, value):
        self.tags[attr[0].capitalize() + attr[1:]] = value
        
    #---------------------------------------------------------------------------
    def __getitem__(self, attr): 
        return self.tags[attr[0].capitalize() + attr[1:]]

    #---------------------------------------------------------------------------
    def secondary_tag_items(self):
        data = self.tags.copy()
        for key in 'White Black FEN'.split():
            data.pop(key)
        
        return sorted([(a,b) for a,b in data.items() if b != '?'])
        
    #---------------------------------------------------------------------------
    @property
    def movetext(self):
        return ''.join([
            '%s%s ' % ('%s.' % (p.move_count,) if p.is_white else '', p.move)
            for p in self.moves
        ])
        
    #----------------------------------------------------------------------------
    def __str__(self):
        tags = dict([(k,'') for k in self.DEFAULT_TAGS])
        tags.update(self.tags)

        s = '\n'.join(
            ['[%s "%s"]' % (k, v.replace('"', '\\"')) for (k,v) in tags.items()]
        )

        if s: 
            s += '\n\n'
            
        movetext = self.movetext
        if movetext: 
            s += '\n'.join(textwrap.wrap(movetext, 80))

        return s + '\n'

    #---------------------------------------------------------------------------
    def annotate(self, annotation):
        self.next_ply.add_annotation(annotation)
        
    #---------------------------------------------------------------------------
    def add_nag(self, data):
        self.next_ply.nag = data
        
    #---------------------------------------------------------------------------
    def __iadd__(self, move):
        if self.is_white:
            self.move_count += 1
            
        self.next_ply = Ply(move, self.is_white, self.move_count)
        self.is_white = not self.is_white
        self.moves.append(self.next_ply)
        return self
                
    #---------------------------------------------------------------------------
    @classmethod
    def parse(cls, text):
        text = text.strip()
        if cls.FEN_RE.match(text):
            pgn = Pgn()
            pgn['FEN'] = text
            return pgn
        else:
            return cls.parse_pgn(text).next()

    #---------------------------------------------------------------------------
    @classmethod
    def parse_pgn(cls, text):
        stream = Stream(text)
        Token = cls.Token
        is_valid = Token.VALID.__contains__
        pgn = cls()
        while True:
            ch = stream.read()
            if not ch:
                break
        
            #--- Ignore whitespace and periods
            if ch in ' .\n\r\t':
                continue

            #--- Discard rest of comment line
            elif ch == Token.COMMENT:
                stream.readline()
                continue

            #--- Process valid symbol tokens
            if ch in Token.VALID:
                symbol = stream.read_tokens(ch, is_valid)
                if not symbol.isdigit():
                    if symbol in pgn.RESULT_STRINGS:
                        pgn['Result'] = symbol
                        yield pgn
                        pgn = cls()
                    else:
                        pgn += symbol

            #--- Start a tag declaration
            elif ch == Token.TAG_START:
                text = stream.read_until(Token.TAG_END)
                while text[-1] == '\\':
                    text = '%s%s' % (
                        text.rstrip('\\'), 
                        stream.read_until(Token.TAG_END)
                    )

                if pgn.move_count:
                    yield pgn
                    pgn = cls()
        
                name, text = text.split(' ', 1)
                pgn[name] = text.strip('"')

            #--- Process descriptive annotation
            elif ch == Token.ANNOTATE_START:
                pgn.annotate(stream.read_until(Token.ANNOTATE_END))
        
            #--- Process recursive annovation variation
            elif ch == Token.RAV_START:
                if not pgn.move_count:
                    raise PgnParseError('Invalid RAV state', stream.index)
            
                pgn.annotate(stream.read_until(Token.RAV_END))

            elif ch == Token.NAG:
                nag = stream.read_tokens(ch, str.isdigit)
        
            else:
                raise PgnParseError('Invalid token [%s:%d]' % (repr(ch), ord(ch)), stream.index)

        yield pgn


#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
class PgnParseError(Exception):

    #---------------------------------------------------------------------------
    def __init__(self, msg, index):
        super(Exception, self).__init__(msg)
        self.offset = index


#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
class ChessError(Exception): 
    '''Base exception class for all chesslib errors.'''
    pass


#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
class ChessMoveError(ChessError): 
    '''Move exceptions.'''
    pass


#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
class CheckingMove(ChessError): 
    '''Last move resulted in checking the opponent's king.'''
    def __init__(self, isMate=False):
        self.isMate = isMate


#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
class CastleError(ChessError):
    '''Castle error at this time.'''


#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
class GameComplete(ChessError): 
    '''No additional moves allowed, game is complete.'''
    pass


#-------------------------------------------------------------------------------
def trace(msg):
    if TRACE:
        f = sys._getframe(1)
        if not isinstance(msg, basestring):
            msg = pprint.format(msg)
            
        sys.stderr.write('TRACE (%s:%d) %s\n' % (f.f_code.co_name, f.f_lineno, msg))


#-------------------------------------------------------------------------------
def is_piece(piece, color=None, type=None):
    '''
    Return true if piece is of type Chesspiece and matches all given criteria.
    '''
    if (
        not isinstance(piece, Chesspiece)      or
        (color and not piece.is_color(color))  or
        (type  and not isinstance(piece,type))
    ):
        return False

    return True


#-------------------------------------------------------------------------------
def opposite_color(color):
    '''
    Returns the constant enumeration of the opposite color.
    '''
    return BLACK if color == WHITE else WHITE


#===============================================================================
class BitFlags:
    '''
    Simple class for handling bitwise manipulation.
    '''
        
    #---------------------------------------------------------------------------
    def __init__(self, value):
        self.value = long(value)
            
    #---------------------------------------------------------------------------
    def clear(self, *args):
        self.value &= ~reduce(operator.or_, args)

    #---------------------------------------------------------------------------
    def match_all(self, *args):
        mask = reduce(operator.or_, args)
        return (self.value & mask) == mask

    #---------------------------------------------------------------------------
    def match_any(self, *args):
        return (self.value & reduce(operator.or_, args)) > 0


#===============================================================================
class Move:

    PROMOTE_TOKEN  = 'QRNB'
    PIECE_TOKEN    = 'QRNBK'
    KCASTLE_TOKEN  = ('O-O', '0-0', 'o-o')
    QCASTLE_TOKEN  = ('O-O-O', '0-0-0', 'o-o-o')
    RESULT_TOKENS  = ('*', '0-1', '1-0', '1/2-1/2', '1/2-0', '0-1/2')
    CAPTURE_TOKEN  = 'x'
    ANNOTATE_TOKEN = '!?'
    
    alt_notation_re = re.compile(r'^([a-h][1-8])[x-]?([a-h][1-8])$')

    #---------------------------------------------------------------------------
    def __init__(self, board, notation):
        self.board       = board
        self.color       = board.color
        self.counter     = board.counter
        self.notation    = notation
        self.capture     = False
        self.checkState  = None
        
        self.piece       = \
        self.castle      = \
        self.result      = \
        self.source      = \
        self.promote     = \
        self.destination = None
        
        self.parse()

    #---------------------------------------------------------------------------
    def parse(self):
        notation = self.notation.rstrip(self.ANNOTATE_TOKEN)
        if notation in self.RESULT_TOKENS:
            self.result = notation
            return

        for check_token in CHECK_TOKENS:
            if notation.endswith(check_token):
                self.checkState = check_token
                notation = notation[:-len(check_token)]
                break

        if notation.startswith(self.KCASTLE_TOKEN):
            self.castle = 2 if notation in self.QCASTLE_TOKEN else 6
            return
            
        m = self.alt_notation_re.match(notation)
        if m:
            self.src = get_position(m.group(1))
            self.dst = get_position(m.group(2))
            self.notation = self.san_notation(get_position(self.src), get_position(self.dst))
            return
         
        if notation[0] in self.PIECE_TOKEN:
            notation, self.piece = notation[1:], notation[:1]

        else:
            if notation[-1] in self.PROMOTE_TOKEN:
                notation, self.promote = notation[:-1], notation[-1:]
                if notation[-1] == '=':
                    notation = notation[:-1]

        notation, self.destination = notation[:-2], notation[-2:]
        if not notation:
            return

        if notation[-1] == self.CAPTURE_TOKEN:
            notation, self.capture = notation[:-1], True

        elif notation[-1] == '-':
            notation = notation[:-1]

        self.source = notation

    #---------------------------------------------------------------------------
    def san_notation(self, src, dest):
        '''
        Convert source/destination notation to algebriac format.
        '''
        self.piece = self.board.get_piece(src.rank, src.file)
        if not is_piece(self.piece):
            raise ChessMoveError, 'No piece at position %s' % (src,)

        if self.piece.color != self.color:
            raise ChessMoveError, 'Illegal move: Moving piece out of turn.'
            
        self.destination = dest
        capture = not self.board.is_empty(dest.rank, dest.file)
        position = self.piece.position

        # check to see if the piece is pawn     
        if isinstance(self.piece, Pawn):
            san = (capture and str(position)[0]) or ''

        else:
            san = self.piece.symbol
    
            # get all similar pieces
            others = self.board.get_pieces(self.piece.color, self.piece.__class__)
            
            # remove our piece
            others.remove(self.piece)
            
            # reduce to remaining pieces that could potentially move here
            others = [n for n in others if dest in n.all_possible_moves()]
            if others:

                # first, check for no piece matching our column         
                if not [o for o in others if o.position.file == position.file]:
                    san += str(position)[0]

                # second, check for no piece matching our rank
                elif not [o for o in others if o.position.rank == position.rank]:
                    san += str(position)[1]

                # otherwise, just use the whole position notation
                else:
                    san += str(position)

        if capture:
            san += 'x'

        san += str(dest)
        if self.checkState:
            san += self.checkState
            
        return san

    #---------------------------------------------------------------------------
    def __repr__(self):
        return 'Move(color=%d, counter=%d, notation="%s")' % (
            self.color, 
            self.counter, 
            self.notation
        )

    #---------------------------------------------------------------------------
    def __str__(self):
        return self.notation
        
    #---------------------------------------------------------------------------
    @property
    def is_white(self):
        return self.color == WHITE
        
    #---------------------------------------------------------------------------
    @property
    def index(self):
        return (self.counter - 1) * 2 + self.color
        


#===============================================================================
class Square:
    '''Chessboard square representive class.'''
    
    #---------------------------------------------------------------------------
    def __init__(self, index):
        self.rank, self.file = index / 8, index % 8

    #---------------------------------------------------------------------------
    def __hash__(self): 
        return self.rank * 8 + self.file

    #---------------------------------------------------------------------------
    def __eq__(self, o):
        return (
            isinstance(o, Square) and 
            self.rank == o.rank and 
            self.file == o.file
        )

    #---------------------------------------------------------------------------
    def __repr__(self):
        return 'Square("%s")' % (str(self))

    #---------------------------------------------------------------------------
    def __str__(self):
        return '%s%s' % (chr(ord('a') + self.file), chr(ord('1') + self.rank))


SQUARES = tuple([Square(i) for i in range(64)])

#-------------------------------------------------------------------------------
def get_position(rank, file=None):
    if isinstance(rank, str):
        file = ord(rank[0]) - 97 # ord('a')
        rank = ord(rank[1]) - 49 # ord('1')

    return SQUARES[rank * 8 + file]


#===============================================================================
class Vector:

    #---------------------------------------------------------------------------
    def __init__(self, piece):
        self.piece = piece
        self.squares = []
        self.guarding = None
        self.checking = False
        self.blocker = None

    #---------------------------------------------------------------------------
    def is_void(self):
        return not (
            self.squares  or 
            self.guarding or 
            self.blocker  or 
            self.checking
        )
        
    #---------------------------------------------------------------------------
    def __str__(self):
        return str(vars(self))

    #---------------------------------------------------------------------------
    def __iadd__(self, square):
        self.squares.append(square)
        return self

    #---------------------------------------------------------------------------
    def guards_piece(self, value=None):
        self.guarding = value or self.guarding
            
        return self.guarding
        
    #---------------------------------------------------------------------------
    def checks_king(self):
        self.checking = self.piece.checking = True


#===============================================================================
class Chesspiece(object):
    '''
    Abstract base class for all piece types.
    '''

    contiguousVector = True

    #---------------------------------------------------------------------------
    def __init__(self, color, board, rank, file):
        '''
        Initialize piece to color, set at position, and associate with board.
        '''
        self.color         = color
        self.board         = board
        self.checking      = False
        self.position      = get_position(rank, file)
        self.oppositeColor = opposite_color(self.color)

        self.board.set_piece(self, self.position.rank, self.position.file)

    #---------------------------------------------------------------------------
    def __repr__(self):
        return '%s%s<%s>' % (
            self.select_if_white('W', 'B'), 
            self.symbol, 
            self.position
        )

    #---------------------------------------------------------------------------
    def __str__(self):
        return '%s%s' % (self.select_if_white(' ', '*'), self.symbol)

    #---------------------------------------------------------------------------
    def __int__(self):
        return self.value
        
    #---------------------------------------------------------------------------
    def select_if_white(self, white, black):
        return white if self.color == WHITE else black

    #---------------------------------------------------------------------------
    def base_name(self):
        '''
        Return the base file name for the image associated with the piece.
        '''
        return '%s%s' % (self.select_if_white('W', 'B'), self.symbol)

    #---------------------------------------------------------------------------
    def is_color(self, color):
        '''
        Return true is piece color matches input.
        '''
        return self.color  == color

    #---------------------------------------------------------------------------
    def reposition(self, position):
        '''
        repositions the piece to square provided.
        '''
        # vacate the old square
        self.board.clear_square(self.position.rank, self.position.file)
        self.position = position
        return self.board.set_piece(self, self.position.rank, self.position.file)

    #---------------------------------------------------------------------------
    def move_to(self, position):
        '''
        Verifies that the position provided is a valid destination before 
        repositioning the piece.
        '''
        
        # validate the destination position
        possibleMoves = self.all_possible_moves()
        if position not in possibleMoves:
            raise ChessError(
                'Move %s: Invalid destination (%s) for %r. Valid moves: [%s]' % (
                    self.board.counter,
                    position, 
                    self, 
                    ','.join([str(m) for m in possibleMoves])
                )
            ) 

        old = self.reposition(position)
        if is_piece(old):
            self.board.halfmove = -1
            
        self.board.ep = '-'
        self.board.halfmoves += 1
        if self.is_color(BLACK):
            self.board.counter += 1

        self.board.color = opposite_color(self.board.color)
        
    #---------------------------------------------------------------------------
    def squares_of_control(self):
        vectors = self.vectors_of_influence()
        if not vectors:
            return vectors
            
        squares = [v.guards_piece().position for v in vectors if v.guards_piece()]
        squares += reduce(lambda x,y: x + y, [v.squares for v in vectors])
        return squares
        
    #---------------------------------------------------------------------------
    def all_possible_moves(self):
        vof = self.vectors_of_influence()
        return vof and reduce(lambda x,y: x + y, [v.squares for v in vof])

    #---------------------------------------------------------------------------
    def vectors_of_influence(self):
        '''
        Return a list of squares along the given vectors that are influence by 
        the piece.
        '''
        self.checking = False
        vectors = []
        
        # walk each position along the paths
        for i, j in self.search_vectors:
            vector = Vector(self)
            currentFile, currentRank = self.position.file, self.position.rank
            blocker = None
            done = False

            while not done:
                currentFile, currentRank = currentFile + i, currentRank + j
                item = self.board.get_piece(currentRank, currentFile)
                
                if item is None:
                    done = True
                
                elif item == EMPTY:
                    if blocker is None:
                        # keep adding squares since path is unobstructed so far
                        vector += get_position(currentRank,currentFile)
                    
                else: 

                    if item.color == self.color:
                        # this is one of our pieces that we are defending: note 
                        # it and stop
                        vector.guards_piece(item)
                        done = True

                    else:
                        # found an enemy
    
                        if blocker is None:
                            # this is the first enemy along our path
                            vector += item.position
                        
                            if isinstance(item, King):
                                item.add_checking_vector(vector)
                                vector.checks_king()
                                done = True
                            
                            else:
                                blocker = item

                        else:
                            # found an enemy, check to see if this is the king;
                            # we can determine later if this current enemy piece 
                            # *could* legally move to block a checking piece; 
                            # otherwise, enemy pieces are just stacked up
                            done = True
                            if not isinstance(item, King):
                                blocker = None


                if not self.contiguousVector:
                    done = True
            
            if blocker is not None:
                vector.blocker = blocker

            if not vector.is_void():
                vectors.append(vector)

        trace('Vectors:\n%s' % (pprint.pformat([str(s) for s in vectors])))
        return vectors


#===============================================================================
class Pawn(Chesspiece):
    '''
    Handles unique pawn behaviour.
    '''

    symbol = 'P'
    value  = 1
    
    #---------------------------------------------------------------------------
    def __init__(self, color, board, rank, file):
        Chesspiece.__init__(self, color, board, rank, file)
        self.direction = self.select_if_white(1, -1)

    #---------------------------------------------------------------------------
    def move_to(self, position):
        # check for en passant capture
        if position == self.board.ep:
            self.board.clear_square(self.position.rank, self.board.ep.file)
        
        # check double move on 1st move
        oldpos = self.position
        Chesspiece.move_to(self, position)
        if abs(oldpos.rank - self.position.rank) == 2:
            self.board.ep = get_position(self.select_if_white(2, 5),oldpos.file)
        
        # finally, check for a promotion
        if self.position.rank in (0,7):
            if not self.board.game.move.promote:
                self.__class__ = CHESS_PIECES['Q']
            else:
                self.__class__ = CHESS_PIECES[self.board.game.move.promote]
            
        self.board.halfmoves = 0

    #---------------------------------------------------------------------------
    def vectors_of_influence(self):
        vectors = []
        self.checking = False
        
        nextRank = self.position.rank + self.direction

        # check for a traditional capture
        for file in (
            self.position.file - self.direction, 
            self.position.file + self.direction
        ):

            if (
                isinstance(self.board.ep, Square) and 
                self.board.ep.rank == nextRank    and 
                self.board.ep.file == file
            ):
                vector = Vector(self)
                vector += self.board.ep
                vectors.append(vector)
                continue

            item = self.board.get_piece(nextRank, file)
            if not is_piece(item, color=self.oppositeColor):
                continue
                
            vector = Vector(self)
            vector += item.position
            if isinstance(item, King):
                vector.checks_king()
                item.add_checking_vector(vector)
                
            vectors.append(vector)

        trace('Vectors: %s' % (pprint.pformat([str(s) for s in vectors])))
        return vectors

    #---------------------------------------------------------------------------
    def squares_of_control(self):
        vectors = []
        self.checking = False
        
        # check for a traditional capture
        for file in (
            self.position.file - self.direction, 
            self.position.file + self.direction
        ):
            item = self.board.get_piece(self.position.rank + self.direction, file)
            if item is None:
                continue

            vector = Vector(self)
            
            if item == EMPTY:
                vector += get_position(self.position.rank + self.direction, file)
                
            elif item.color == self.oppositeColor:
                vector += item.position
                if isinstance(item, King):
                    vector.checks_king()
                    item.add_checking_vector(vector)

            else:
                vector.guards_piece(item)
                
            vectors.append(vector)

        trace('Vectors:\n%s' % (pprint.pformat([str(s) for s in vectors])))
        squares = []
        if vectors:
            squares = [v.guards_piece().position for v in vectors if v.guards_piece()]
            squares += reduce(lambda x,y: x + y, [v.squares for v in vectors])
        
        return squares


    #---------------------------------------------------------------------------
    def all_possible_moves(self):
        possible = Chesspiece.all_possible_moves(self)

        # check for a first move of 2 squares
        nextRank1 = self.position.rank + self.direction
        nextRank2 = nextRank1 + self.direction
        if self.select_if_white(1, 6) == self.position.rank:
            if (
                self.board.is_empty(nextRank1, self.position.file) and 
                self.board.is_empty(nextRank2, self.position.file)
            ):
                possible.append(get_position(nextRank2, self.position.file))

        # check the next rank
        if self.board.is_empty(nextRank1, self.position.file):
            possible.append(get_position(nextRank1, self.position.file))
            
        return possible
        
    
#===============================================================================
class Rook(Chesspiece):
    '''
    Handles unique rook behaviour.
    '''

    symbol = 'R'
    value  = 5
    search_vectors = ORTHOGONAL_VECTORS

    #---------------------------------------------------------------------------
    def move_to(self, position):
        '''
        First clears the option for this rook to participate in castling.
        '''
        if self.position.file in (0,7):
            state = self.select_if_white(CASTLE_WQ, CASTLE_BQ)
            if self.position.file == 7:
                state = self.select_if_white(CASTLE_WK, CASTLE_BK)

            self.board.castle.clear(state)

        Chesspiece.move_to(self, position)


#===============================================================================
class Knight(Chesspiece):
    '''
    Handles unique knight behaviour.
    '''

    symbol = 'N'
    value  = 3
    contiguousVector = False
    search_vectors = (
        ( 2, 1), 
        ( 2,-1), 
        (-2, 1), 
        (-2,-1), 
        ( 1, 2), 
        ( 1,-2), 
        (-1, 2), 
        (-1,-2)
    )


#===============================================================================
class Bishop(Chesspiece):
    '''
    Handles unique bishop behaviour.
    '''

    symbol = 'B'
    value  = 3
    search_vectors = DIAGONAL_VECTORS


#===============================================================================
class Queen(Chesspiece):
    '''
    Handles unique queen behaviour.
    '''

    symbol = 'Q'
    value  = 9
    search_vectors = DIAGONAL_VECTORS + ORTHOGONAL_VECTORS


#===============================================================================
class King(Chesspiece):
    '''Handles unique king behaviour.'''
    
    symbol = 'K'
    value  = 0
    search_vectors = DIAGONAL_VECTORS + ORTHOGONAL_VECTORS
    contiguousVector = False

    #---------------------------------------------------------------------------
    def __init__(self, color, board, rank, file):
        Chesspiece.__init__(self, color, board, rank, file)
        self.castleBits  = self.select_if_white(CASTLE_W, CASTLE_B)
        self.castleKBits = self.select_if_white(CASTLE_WK, CASTLE_BK)
        self.castleQBits = self.select_if_white(CASTLE_WQ, CASTLE_BQ)
        self.castleOps = ((self.castleKBits, 6), (self.castleQBits, 2))

        self.checkingVectors = []

    #---------------------------------------------------------------------------
    def move_to(self, position):
        '''First check for a castling move, otherwise move normally. 
        Sets castling flags to invalid.'''
        if self.position.file == 4 and position.file in (2,6):
            self._castle(position.file)
        else:
            Chesspiece.move_to(self, position)

        self.board.castle.clear(self.castleBits)

    #---------------------------------------------------------------------------
    def add_checking_vector(self, vector):
        self.checkingVectors.append(vector)

    #---------------------------------------------------------------------------
    def _get_invalid_squares(self):
        '''
        Return a dictionary containing all squares that attacked by enemy pieces.
        The dictionary is keyed by squares with values of None.
        '''
        
        # look for direct check
        invalidSquares = {}
        for piece in self.board.get_pieces(self.oppositeColor):

            squares = piece.squares_of_control()
            
            for sq in squares:
                invalidSquares[sq] = None

        #trace('Invalid squares [%s]: %s' % (self, invalidSquares.keys()))
        return invalidSquares
                
    #---------------------------------------------------------------------------
    def get_check_state(self):
        '''
        Returns the appropriate value if the king is in check, mated, or not.
        '''
        
        self.checkingVectors = []

        # get the pieces that are checking the king (this could be two)
        invalidSquares = self._get_invalid_squares()

        countOfAttackers = len(self.checkingVectors)
        if countOfAttackers == 0:               
            return None
            
        #trace('King is in check (Move #%d' % self.board.counter)
            
        # First check to see if the king can move out of check
        for move in Chesspiece.all_possible_moves(self):
            if move not in invalidSquares:
                #trace('King can move to %s' % move)
                #trace('Invalid moves: %s' % invalidSquares.keys())
                return CHECK_TOKEN

        #trace('King cannot move out of check (%d attackers)' % countOfAttackers)
        if countOfAttackers == 2:
            #trace('King attacked by multiple sources: Mate')
            return CHECKMATE_TOKEN
            
        attackVector = self.checkingVectors[0]
        defendables = [attackVector.piece.position] + attackVector.squares
        defendables.remove(self.position)
        
        #trace('Defendable squares: %s ' % defendables)
        
        for defender in self.board.get_pieces(self.color, *ATTACKERS):
            for square in defender.all_possible_moves():
                if square in defendables:
                    #trace('King defendable by %s at %s' % (defender, square))
                    return CHECK_TOKEN

        return CHECKMATE_TOKEN
        
    #---------------------------------------------------------------------------
    def _castle(self, destCol):
        '''
        Performs all verification for castling as well as both king and rook moves.
        '''
        # verify that the King has not moved yet
        if not self.board.castle.match_any(self.castleBits):
            raise CastleError, 'Castling not allowed.'

        # verify that the king is trying to castle out of check
        invalidSquares = self._get_invalid_squares()
        if self.position in invalidSquares:
            raise CastleError, 'King cannot castle out of check'

        # check to see that there is a rook in the correct position
        kingSide = destCol > self.position.file
        rook = self.board.get_piece(self.position.rank, kingSide and 7 or 0)
        if not is_piece(rook, self.color, Rook):
            raise CastleError, 'Piece is not Rook'

        # get the castling direction            
        state = (kingSide and self.castleKBits) or self.castleQBits

        if not self.board.castle.match_any(state):
            raise CastleError(
                'Cannot castle %s-side' % (kingSide and 'king' or 'queen')
            )

        # evaluate the destination and transition squares
        for i in range(destCol, self.position.file, cmp(self.position.file,destCol)):
            square = get_position(self.position.rank, i)
            
            # make sure the square is empty
            if not self.board.is_empty(square.rank, square.file):
                raise CastleError, 'Blocking piece at %s' % square
                
            # check for moving through check
            if square in invalidSquares:
                raise CastleError, 'Moving through check at %s' % square

        # if castling queenside, make sure b column is empty
        if not kingSide and not self.board.is_empty(self.position.rank,1):
            raise CastleError, 'Blocking piece at %s' % square
        
        position = get_position(
            self.position.rank, 
            destCol > self.position.file and 5 or 3
        )
        rook.move_to(position)
        
        # move the king
        destPos = get_position(self.position.rank, destCol)
        self.board.game.move.notation = 'O-O'
        if destCol < self.position.file:
            self.board.game.move.notation += '-O'
            
        self.reposition(destPos)
        return destPos

    #---------------------------------------------------------------------------
    def all_possible_moves(self):
        possible = Chesspiece.all_possible_moves(self)

        # check to see if the king can castle
        for side, file in self.castleOps:
            if self.board.castle.match_any(side):
                possible.append( get_position(self.position.rank, file) )

        return possible


    #---------------------------------------------------------------------------
    def vectors_of_influence(self):
        '''
        Return a list of squares along the given vectors influenced by the piece.
        '''
        vectors = []
        
        # walk each position along the paths
        for i, j in self.search_vectors:
            file, rank = self.position.file + i, self.position.rank + j
            item = self.board.get_piece(rank, file)
            if item is None:
                continue
            
            vector = Vector(self)
            if item == EMPTY:
                vector += get_position(rank, file)
            else: 
                if item.color == self.color:
                    # this is one of our pieces that we are defending: note it 
                    # and stop
                    vector.guards_piece(item)
                else:
                    # found an enemy
                    vector += item.position
                    
            vectors.append(vector)

        trace('Vectors:\n%s' % (pprint.pformat([str(s) for s in vectors])))
        return vectors


CHESS_PIECES = dict(
    K=King,
    Q=Queen,
    R=Rook,
    N=Knight,
    B=Bishop,
    P=Pawn
)

ATTACKERS = (Queen, Rook, Knight, Bishop, Pawn)

#-------------------------------------------------------------------------------
def castle_parser(x, y):
    return x | CASTLE_MAP.get(y,0)



#===============================================================================
class Chessboard:
    '''Handles chess board behavior.'''

    BLACK_CHARS = [key.lower() for key in CHESS_PIECES.keys()]

    #---------------------------------------------------------------------------
    def __init__(self, game, fenNotation=None):
        self.game = game
        self.set_board_to_fen(fenNotation or Pgn.DEFAULT_FEN)

    #---------------------------------------------------------------------------
    def parse_fen(self, fenString):
        '''
        Seperate the FEN notation into components and initialize state variables:

        * Piece placement
        * Active color (w|b)
        * Castling availability (KQkq-)
        * En passant target square ([a-h][1-8]|-)
        * Halfmoves - count ply since last pawn advance or capturing move (50 max)
        * Fullmove number - increments after black's move
            
        '''
            
        self.placement, self.color, castle, ep, halfmoves, counter = fenString.split()
        self.halfmoves = int(halfmoves)
        self.counter = int(counter)
        self.ep = ep != '-' and get_position(ep) or ep
        self.castle = BitFlags(reduce(castle_parser, castle, 0))
        
    #---------------------------------------------------------------------------
    def set_board_to_fen(self, fenNotation):
        '''
        Parse the FEN notation and set the board and game state accordingly.
        '''

        self.parse_fen(fenNotation)
        self.board = {}
        rank, file = 7, 0
        for i in range(len(self.placement)):
            if file > 7:
                rank, file = rank - 1, 0
                if rank < 0:
                    raise ChessError, 'Invalid state near ' + self.placement[i:]
                    
            if self.placement[i].isdigit():
                file += int(self.placement[i])
            
            elif self.placement[i] != '/':
                color = WHITE
                key = self.placement[i]
                if key in self.BLACK_CHARS:
                    key = key.upper()
                    color = BLACK

                piece = CHESS_PIECES.get(key)
                if piece is None:
                    raise ChessError, 'Invalid FEN notation: ' + self.placement[i]
                    
                piece(color, self, rank, file)
                file += 1

    #---------------------------------------------------------------------------
    def clear_square(self, rank, file):
        key = (rank,file)
        if key in self.board:
            del self.board[key]

    #---------------------------------------------------------------------------
    def is_empty(self, rank, file):
        return EMPTY == self.get_piece(rank, file)

    #---------------------------------------------------------------------------
    def get_king(self, color):
        return self.get_pieces(color, King)[0]

    #---------------------------------------------------------------------------
    def get_pieces(self, color, *types):
        types = types or Chesspiece
        return [
            p for p in self.board.values() 
            if isinstance(p, types) and p.color == color
        ]

    #---------------------------------------------------------------------------
    def get_piece(self, rank, file):
        if 0 > rank or rank > 7 or 0 > file or file > 7:
            return None

        return self.board.get((rank,file), EMPTY)

    #---------------------------------------------------------------------------
    def set_piece(self, item, rank, file):
        if 0 > rank or rank > 7 or 0 > file or file > 7:
            raise ChessError, 'Board position out of range'

        key = (rank, file)
        olditem = self.board.get(key, EMPTY)
        self.board[key] = item
        return olditem

    #---------------------------------------------------------------------------
    def get_fen_placement_string(self):
        '''Generates a FEN placement string from the current position.'''
        ps = ''
        files = [0, 1, 2, 3, 4, 5, 6, 7]
        for rank in [7, 6, 5, 4, 3, 2, 1, 0]:
            if ps:
                ps += '/'
                
            empty = 0
            for file in files:
                pcs = self.board.get((rank,file))
                if pcs:
                    if empty:
                        ps += str(empty)
                        empty = 0

                    ps += pcs.symbol if pcs.color == WHITE else pcs.symbol.lower()
                else:
                    empty += 1
            if empty:
                ps += str(empty)
        return ps

    #---------------------------------------------------------------------------
    def get_fen_notation(self):
        clr = 'w' if self.color == WHITE else 'b'
        cstl = ''
        for bit, s in CASTLE_FORMAT:
            if self.castle.match_all(bit):
                cstl += s
        
        return ' '.join([ 
            self.placement, 
            clr,
            cstl or '-',
            str(self.ep),
            str(self.halfmoves),
            str(self.counter)
        ])
        
    #---------------------------------------------------------------------------
    def __str__(self):
        border = '  +----+----+----+----+----+----+----+----+\n'
        strrep = border
        for j in range(7, -1, -1):
            rank = [self.get_piece(j, i) for i in range(8)]
            rankStr = ' | '.join(['%s' % file for file in rank])
            strrep += '%d | %s |\n%s' % (j + 1, rankStr, border)

        return strrep + '     %s\n' % ('    '.join(list('abcdefgh')))


#===============================================================================
class Game(object):
    '''
    Chess game interfaces
    '''

    #---------------------------------------------------------------------------
    def __init__(self, pgn=None):
        if isinstance(pgn, basestring):
            pgn = Pgn.parse(pgn)
            
        self.pgn = pgn or Pgn()
        self.fen = [self.pgn['FEN']]
        self.result = '*' #self.pgn.GetResult()
        self.board = Chessboard(self, self.fen[0])
        self.move = None
        self.moves = []
        self.can_continue = True

    #---------------------------------------------------------------------------
    def to_json(self):
        def move_json(m):
            return {
                'fen':        m.fen,
                'notation':   m.notation,
                'annotation': ' | '.join(m.annotations),
                'src':        str(m.origin),
                'dst':        str(m.destination)
            }
            
        return json.dumps({
            'info': {
                'white': self.pgn['white'],
                'black': self.pgn['black'],
                'extra': self.secondary_tag_items()
            },
            'moves': [{
                'fen': self.fen[0],
                'notation': '',
                'annotation': ''
            }] + [move_json(m) for m in self.moves]
        }, indent=4)
        
    #---------------------------------------------------------------------------
    def __str__(self):
        return str(self.board)

    #---------------------------------------------------------------------------
    def secondary_tag_items(self):
        return [[a,b] for a,b in self.pgn.secondary_tag_items()]
        
    #---------------------------------------------------------------------------
    def move_exposes_check(self, king):
        '''
        Verify that move leading to current position didn't expose check 
        '''
        
        for attacker in self.board.get_pieces(
            king.oppositeColor, 
            Rook, 
            Bishop, 
            Queen
        ):
            if king.position in attacker.all_possible_moves():
                return True
                
        return False
                    
    #---------------------------------------------------------------------------
    def undo_move(self):
        '''Roll back the state of the game to previous position.'''
        if len(self.fen) > 1:
            bad = self.fen.pop(-1)
            self.board.set_board_to_fen(self.fen[-1])
            return self.fen[-1]

    #---------------------------------------------------------------------------
    def eval_san_notation(self):
        if isinstance(self.move.piece, Chesspiece):
            return
            
        if self.move.piece:
            self.move.all = self.board.get_pieces(
                self.move.color, 
                CHESS_PIECES[self.move.piece]
            )

        else:
            pawns = self.board.get_pieces(self.move.color, Pawn)
            whichFile = ord((self.move.source or self.move.destination)[0]) - 97
            self.move.all = [p for p in pawns if p.position.file == whichFile]

        if not self.move.all:
            raise ChessMoveError, 'Unable to determine which piece to move'
            
        self.move.destination = get_position(self.move.destination)
        if len(self.move.all) == 1:
            self.move.piece = self.move.all[0]

        else:
            self.move.source = self.move.source or ''

            self.move.piece = None
            for piece in self.move.all:
                if  0 <= str(piece.position).find(self.move.source) \
                and self.move.destination in piece.all_possible_moves():
                    if self.move.piece is not None:
                        raise ChessMoveError, 'Ambiguous Move'

                    self.move.piece = piece

            if self.move.piece is None:
                raise ChessMoveError, 'No valid piece available:\n%s' % (
                    pprint.pformat(vars(self.move))
                )
        
    #---------------------------------------------------------------------------
    def make_move(self, notation, annotations=None):
        if not self.can_continue:
            raise ChessMoveError(
                'Illegal move (%s): Game cannot continue!' % (notation,)
            )

        try:
            self.move = Move(self.board, notation)
            if self.move.result:
                raise GameComplete(self.move.result)

        except GameComplete, result:
            self.result = str(result)
            return None

        try:
            if self.move.castle:
                self.move.piece = self.board.get_king(self.move.color)
                self.move.origin = self.move.piece.position
                self.move.destination = get_position(
                    self.move.piece.position.rank, 
                    self.move.castle
                )

                self.move.piece.move_to(self.move.destination)

            else:
                self.eval_san_notation()
                self.move.origin = self.move.piece.position
                self.move.piece.move_to(self.move.destination)

                if self.move_exposes_check(self.board.get_king(self.move.color)):
                    raise ChessMoveError, 'Move results with king in check'
            
        except ChessError, why:
            print self.board
            if isinstance(why, ChessError):
                self.undo_move()
                why.args += (' [%r]' % (self.move, ),)
                raise
            else:
                typ, val, tb = sys.exc_info()
                raise ChessError('[%r] %s' % (self.move, val))
            
        king = self.board.get_king(opposite_color(self.move.color))
        checkState = king.get_check_state()
        if checkState == CHECK_TOKEN:
            if '+' not in self.move.notation:
                self.move.notation += '+'

        elif checkState == CHECKMATE_TOKEN:
            self.result = self.move.piece.select_if_white('1-0', '0-1')
            if self.move.notation[-1] != '#':
                self.move.notation += '#'
                self.can_continue = False

        self.board.placement = self.board.get_fen_placement_string()
        self.move.result = self.result
        self.move.fen = self.board.get_fen_notation()
        self.move.annotations = annotations or []
        self.fen.append(self.move.fen)
        self.moves.append(self.move)
        return self.move
        
    #---------------------------------------------------------------------------
    def play(self, cb=None, debug=False):
        for ply in self.pgn:
            if debug:
                print ply
            move = self.make_move(ply.move, ply.annotations)
            if move is None:
                return
                
            if debug:
                print move.origin, move.destination
                print self
                
            ply.move = move.notation
            if cb and callable(cb):
                cb(move)
            


#-------------------------------------------------------------------------------
def play_game(history):
    game = Game(Pgn.Read(history))
    if history:
        game.play()

    while 1:
        print '\n%s\n' % str(game)
        next = raw_input('Next Move> ')
        if not next: 
            break
        elif next == 'u':
            game.undo_move()
        else:
            move = game.make_move(next)
            print move.notation


#-------------------------------------------------------------------------------
def test_pgn():
    '''1. c4 Nf6 2. Nf3 d5 3. cxd5 Nxd5 4. Nc3 Bf5 5. Qa4+ Nc6 6. Ne5 Ndb4 7. Nxc6 Nxc6 8. e4 Bg6 9. Bb5 Qd7 10. d4 O-O-O 11. d5 Bxe4 12. Bxc6 1-0'''
    import sys
    from datetime import datetime
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else test_pgn.__doc__
    try:
        start = datetime.now()
        games = list(Pgn.parse_pgn(text))
        delta1 = datetime.now() - start
        delta1 = float(delta1.seconds * 1000 + delta1.microseconds / 1000)
        print delta1
        print len(games)
        print str(games[0])
    except Exception, why:
        print why
        raise


#-------------------------------------------------------------------------------
def test(move_text, *next_moves):
    game = Game(Pgn.parse(move_text))
    game.play(debug=False)
    for move in next_moves:
        game.make_move(move)
        
    print 
    print game
    linelen = 0
    for move in game.moves:
        notation = move.notation
        s = '%s. %s' % (move.counter, notation) if move.is_white else notation
        print s,
        linelen += len(s) + 1
        if linelen >= 44:
            linelen = 0
            print
            
    print


################################################################################
if __name__ == '__main__':
    import sys
    text = '1. c4 Nf6 2. Nf3 d5 3. cxd5 Nxd5 4. Nc3 Bf5 5. Qa4+ Nc6 6. Ne5 Ndb4 7. Nxc6 Nxc6 8. e4 Bg6 9. Bb5 Qd7 10. d4 O-O-O 11. d5 Bxe4 12. Bxc6 1-0'
    args = sys.argv[1:]
    if args and args[0] == '-d':
        args.pop(0)
        import pdb; pdb.set_trace()
    
    args = [open(a).read() for a in args] or [text]
    for arg in args:
        test(arg)
