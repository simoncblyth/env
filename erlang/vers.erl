-module(vers).

-export([start/0]).


%%%  wonderful ... this dont work with the old erlang on C ... the otp_release atom not defined ??
start() -> 
 %%% io:format(" otp_release ~p version ~p ~n ", [erlang:system_info(otp_release),erlang:system_info(version)]).
 io:format("version ~p ~n", [erlang:system_info(version)]).

