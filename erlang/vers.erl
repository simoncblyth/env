-module(vers).

-export([start/0]).

start() -> 
 io:format(" otp_release ~p version ~p ~n ", [erlang:system_info(otp_release),erlang:system_info(version)]).

