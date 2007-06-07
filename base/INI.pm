package INI ;
use Data::Dumper ;

=cut 

   edits ini files , preserves comments

   started with Config::INI::Simple from CPAN , but found deficiencies (new
   qtys could not be targeted to a specific block ) hence I rolled my own 

  ./ini-edit.pl trac.ini block1:val1:qwn1 block1:val2:qwn block2:val3:qwn
  .... block block1 key val1 val qwn1 
  .... block block1 key val2 val qwn 
  .... block block2 key val3 val qwn 
  
   diff trac.ini trac.ini.out
	  114a115,122
	  > 
	  > [block1]
	  > val2 = qwn
	  > val1 = qwn1
	  > 
	  > [block2]
	  > val3 = qwn
	  > 

    Issues:
	    command line block order (and qty order) is not preserved in output 

=cut



sub EDIT {

   my ( $file , @edits ) = @_ ;
   
    print "INI::EDIT reading $file \n" ; 
   my $ini = new INI ;
   $ini->read( $file );
   $ini->edit( @edits );
   $ini->prepare();
   
   #$ini->write("$file.out" );
   $ini->write("$file" );
   print Dumper( $ini );
}


sub new{
    my $proto = shift;
	my $class = ref($proto) || $proto || 'CONF';

	$self->{'data'} = {} ;
	$self->{'blockorder'} = [] ;
	bless ($self,$class);
	return $self;
}

sub read{

   my ($self, $file ) = @_ ;

   open(F,"<$file") || die "cannot open $file\n " ;
   my @lines = <F> ;
   chomp @lines ;
   close F ;

   my $data = {} ;
   my $block = "___start___" ;

   for my $line (@lines){
      ## block specifier or content  	
      if ( $line =~ m/\[(\S*)\]/ ){
         $block = $1 ;
	     push( @{ $self->{'blockorder'} } , $block );
	     $self->{'blockline'}{$block} = $line ;
		 
      } else {
	      push( @{ $self->{'data'}{$block}{'lines'} }, $line );
		  my $pair = &interpline( $line );
		  if( $#{ $pair } + 1 == 2 ){
		     push( @{ $self->{'data'}{$block}{'keyorder'} }, ${ $pair }[0] ) ;
		     $self->{'data'}{$block}{'content'}{${ $pair }[0]} = ${ $pair }[1] ;
          } 
       }   ## block or otherwise
    }	
}


sub interpline{

    my ( $line ) = @_ ;
    if( length $line == 0 || $line =~ /^\s*\;/ || $line =~ /^\s*\#/ ){ 
	    return [] ;
	} elsif( $line =~ m/\s*(\S*)\s*=\s*(.*)\s*$/ ){
		return [$1,$2] ;
    } else {
        print "ERROR unhandled line $line \n ";
		return [] ;
	}
}

sub formline{
	my ( $key , $val ) = @_ ;
	return  sprintf("%s = %s", $key, $val );
}

sub formblock{
	my ( $block ) = @_ ;
	return  sprintf("[%s]", $block );
}


sub prepare{

   push(@{ $self->{'edit'} }, $_) for (@{ $self->{'data'}{'___start___'}{'lines'} });

   for my $block (@{ $self->{'blockorder'} }){
       
       print "preparing block $block \n";
       push(@{ $self->{'edit'} }, $self->{'blockline'}{$block} );

       for my $line (@{ $self->{'data'}{$block}{'lines'} }){

		 # print "$line\n"; 
          my $pair = &interpline( $line );
		  if( $#{ $pair } + 1 == 2 ){
			  
		       my $key = ${ $pair }[0] ;
			   my $val = ${ $pair }[1] ;
               
			   ## no change in content, use original line ... otherwise form newline
			   if( $self->{'data'}{$block}{'content'}{$key} eq $val ){
			      push(@{ $self->{'edit'} }, $line); 
               } else {
			      push(@{ $self->{'edit'} }, &formline( $key, $val )); 
			   }

               ## check for last(at input) key in the block 
			   if( $key eq ${ $self->{'data'}{$block}{'keyorder'} }[-1] ){
                  $self->addkeys( $block );
				}
			
          } else {
              push(@{ $self->{'edit'} }, $line ); ## pass thru comments untouched
		  }

	   }	   

   }

   ## check for new blocks
 
    my $lastblock = ${ $self->{'blockorder'} }[-1] ;
	my $lastline = ${ $self->{'data'}{$lastblock}{'lines'} }[-1] ;
 
    print "checking for new blocks ... lastblock $lastblock lastline $lastline \n" ;
    
    my $nadd = 0 ;
    for my $addblock (keys %{ $self->{'data'} }){
        
        
    
       if( grep( $addblock eq $_, @{ $self->{'blockorder'} }) == 0 && $addblock ne "___start___" ){

           push(@{ $self->{'edit'} }, "# new block $addblock  " ); 
           ## add a blank line before the new block if one not there already  
           ++$nadd ; 
		   if( $nadd == 1 && $lastline ne "" ){
             push(@{ $self->{'edit'} }, "" ); 
		   }

           
		   
           push(@{ $self->{'edit'} }, &formblock( $addblock) ); 
           $self->addkeys( $addblock );
           push(@{ $self->{'edit'} }, "" ); 
	   } else {
           push(@{ $self->{'edit'} }, "#not new block $addblock  " );
	   }
    }

}


sub write{
    my ( $self , $file ) = @_ ;
    open(F,">$file");
    for my $line (@{ $self->{'edit'} }){
      printf F "%s\n", $line ; 
    }
    close F;
}


sub addkeys {

   my ( $self , $block ) = @_ ;
   #push(@{ $self->{'edit'} }, "#last key" );
   for my $addkey (keys %{ $self->{'data'}{$block}{'content'} }){
        ## check for new keys
		if( grep( $addkey eq $_, @{ $self->{'data'}{$block}{'keyorder'} }) == 0 ){
			 my $addval = $self->{'data'}{$block}{'content'}{$addkey} ;
			 push(@{ $self->{'edit'} }, &formline( $addkey, $addval )); 
		} else {
		     #push(@{ $self->{'edit'} }, "#not new key $addkey " );
		}
	}
}



sub edit{

   my ( $self , @args ) = @_ ;
   my ($block,$key,$val) ;
   for (@args){
		($block,$key,$val) = split /:/ ;
		print "  .... block $block key $key val $val \n" ;
		$self->{'data'}{$block}{'content'}{$key} = $val;
   }

}



	
1;
